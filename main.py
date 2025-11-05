import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import logging
from datetime import datetime
import numpy as np
import os

from model import DataGen, Encoder, Decoder
from utils.utils import *

from config import parser

# ============================ pre-process ============================ 
# load configs.
args = parser.parse_args()

# set random seeds.
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# log outputs.
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
# include key hyperparameters in log filename.
hyperparams_str = f"embd{args.embd_dim}_maxlen{args.max_length}_lam{args.lam:.0e}_pe{args.p_e:.0e}"
log_filename = f'logs/result_{hyperparams_str}_{current_time}.log'
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(log_filename)
format_str = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(log_filename)
stream_handler.setFormatter(format_str)
file_handler.setFormatter(format_str)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# set cuda.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# load datasets.
d = DataGen(args.dataset)

# initialize E2EC en/decoders.
en = Encoder(args.input_dim, args.embd_dim, args.max_length).to(device)
de = Decoder(args.embd_dim, args.output_dim, args.max_length).to(device)

# define optimizer for encoder and decoder.
opt = torch.optim.Adam(en.param()+de.param(), lr= args.lr, weight_decay= 1e-3)

# load model if specified.
if args.load_model is not None:
    load_path = args.load_model
    if not os.path.isabs(load_path):
        load_path = os.path.join(args.model_dir, load_path)
    success = load_model(en, de, opt, load_path, device, logger)
    if not success:
        logger.error("Failed to load model. Exiting.")
        exit(1)

# ============================ training ============================ 
elif args.train:
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        for b, (x, y) in enumerate(d.batch(batch_size=args.bs)):
            x = x.to(device)
            y = y.to(device)
            x = x.view(x.size(0), -1)
            
            # encode: length l, content zt, continuous latent code zc, 
            # and their respective distributions p_l, p_zt, p_zc.
            l, p_l, zt, p_zt, zc, p_zc = en.encode(x)
            
            # truncate to variable-length code.
            z = truncate(zt, l)

            # channel.
            zh = channel(z, p_e=args.p_e)
            
            # decoding.
            yh, xh = de.decode(zh)

            # rate-distortion tradeoff:  D + lam * R.
            # D: semantic distortion over a noisy channel.
            # R: rate, a.k.a. average length of code.
            D = F.cross_entropy(yh, y, reduction='none')
            R = avg_len(p_l)

            # regularization terms w.r.t. a continuous counterpart of E2EC,
            # for convergence speedup and numerical stability.
            # MI: mutual information of continuous ver. 
            # Dc: semantic distortion of continuous ver.
            # KD: symbol grounding technique as T. Lin's NIPS 2021:
            # "learning to ground multi-agent communication with autoencoders", 
            # a.k.a. a type of knowledge distillation.
            # H: entropy of length distribution, entropy regularization for policy gradient.
            yhc = de.decode_c(zc)
            MI = MI_est(zc, p_zc)
            Dc = F.cross_entropy(yhc, y, reduction='none')
            KD = F.kl_div(
                F.log_softmax(yh / args.T, dim=1), 
                F.softmax(yhc.detach() / args.T, dim=1), reduction='batchmean'
            ) + 1e-2*F.mse_loss(xh, zc.detach())
            H = p_l.entropy().mean() + p_zt.entropy().mean()
            reg = Dc.mean() + KD + args.lam*MI - 1e-2*H

            # loss: main objective, i.e., the regularized rate-distortion tradeoff.
            loss = D.mean() + args.lam*R + args.scal*reg
            
            # policy gradient for non-differentiable operations with variance reduction.
            # adv: advantage, i.e., distortion - baseline.
            # logp: log-probability for both length and content sampling.
            # logl: log-likelihood.
            adv = D.detach() - Dc.detach()
            logp = p_l.log_prob(l) + p_zt.log_prob(zt).sum(dim=1)
            logl = torch.mean(logp * adv)

            # gradient backward for loss and log likelihood.
            bkwd_val = loss + logl
            bkwd_val.backward()

            # update parameters.
            if b % args.update_freq == 0:
                if args.grad_clip is not None:
                    clip_grad_norm_(en.param() + de.param(), args.grad_clip)
                opt.step()
                opt.zero_grad()
            
            # record info.
            if b % args.log_freq == 0:
                with torch.no_grad(): Var = avg_variance(p_l)
                info_output = (
                    f"Epoch {epoch}, Batch {b}, logl: {logl:.4f}, Loss: {loss:.4f}, "
                    f"D: {D.mean():.4f}, R: {R:.4f}, MI: {MI:.4f}, Dc: {Dc.mean():.4f}, "
                    f"KD: {KD:.4f}, Var: {Var:.4f}"
                )
                logger.info(info_output)

        # save model if specified.
        if args.save_model is not None and epoch % args.eval_freq == 0:
            # create hyperparameter suffix for filename.
            hpsfix = f"_embd{args.embd_dim}_maxlen{args.max_length}_lam{args.lam:.0e}_pe{args.p_e:.0e}"
            save_path = args.save_model
            if not os.path.isabs(save_path): save_path = os.path.join(
                args.model_dir, f"{save_path}{hpsfix}_epoch_{epoch}.pth"
            )
            else: save_path = f"{save_path}{hpsfix}_epoch_{epoch}.pth"
            save_model(en, de, opt, epoch, loss.item(), save_path, logger)
            ckpt = torch.load(save_path, map_location=device)
            ckpt['args'] = args
            ckpt['hyperparams'] = {
                'embd_dim': args.embd_dim, 'max_length': args.max_length, 'lam': args.lam, 
                'p_e': args.p_e, 'lr': args.lr, 'bs': args.bs, 'epochs': args.epochs, 'seed': args.seed
            }
            torch.save(ckpt, save_path)
            
    logger.info("Training completed.")

# ============================ evaluation ============================ 
if args.eval:
    logger.info("Starting evaluation...")
    eval_model(en, de, d, device, logger, args.bs, args.p_e)
