import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np


def MI_est(z, p):
    """
    asymptotic unbiased estimation of mutual information I(X;Z) for variable-length codes
    uses a simplified approach that considers only valid (non-truncated) positions
    """
    ps_list = list()
    for k in range(z.size(0)):
        ps_list.append(
            torch.roll(torch.exp(
                p.log_prob(torch.roll(z, shifts= -k, dims= 0))
            ), shifts= k, dims= 0)
        )
    ps = torch.stack(ps_list)
    log_pz = torch.log(torch.mean(ps, dim= 0))

    return torch.mean(torch.sum((p.log_prob(z) - log_pz), dim= -1))


def avg_len(p_l):
    """
    calculate average length from length distribution
    """
    # p_l is a categorical distribution
    probs = p_l.probs
    lengths = torch.arange(probs.size(1), device=probs.device, dtype=torch.float)
    return torch.sum(probs * lengths, dim=1).mean()


def avg_variance(p):
    """
    calculate variance corresponding to length distribution
    """
    pbars = p.probs.mean(dim=0)
    pbars = pbars / pbars.sum()

    pbars = torch.clamp(pbars, min=1e-10)

    n = pbars.shape[0]
    indices = torch.arange(n, dtype=torch.float32, device=pbars.device)
    m = (pbars * indices).sum()
    m_sq = (pbars * indices**2).sum()
    
    return m_sq - m**2


def channel(z, p_e=0.1):
    """
    binary symmetric channel (BSC):
    > generate Bernoulli noise of dimension Rmax (full z dimension)
    > apply element-wise binary addition (XOR) between z and noise
    > truncate result using -1 markers for variable length
    """
    bs, max_length = z.shape
    device = z.device
    
    # generate Bernoulli noise for full Rmax dimension
    noise = torch.bernoulli(torch.full((bs, max_length), p_e, device=device))
    
    # apply element-wise binary addition (XOR)
    # for valid positions (not -1), XOR with noise
    # for truncated positions (-1), keep as -1
    z_hat = z.clone()
    
    # create mask for valid positions (not -1)
    valid_mask = z != -1
    
    # convert to integer for XOR operation (XOR only works with integer types)
    z_int = z[valid_mask].long()
    noise_int = noise[valid_mask].long()
    
    # apply XOR to valid positions
    z_hat[valid_mask] = (z_int ^ noise_int).float()
    
    return z_hat


def truncate(z_tilde, l):
    """
    truncate z_tilde to length l
    use -1 to mark truncated regions to distinguish from valid 0-1 code flow
    """
    bs = z_tilde.size(0)
    max_length = z_tilde.size(1)
    truncated = torch.full_like(z_tilde, -1)  # initialize with -1 for truncated regions
    
    for i in range(bs):
        length = min(l[i].item(), max_length) + 1
        truncated[i, :length] = z_tilde[i, :length]  # keep actual 0-1 values for valid length
        
    return truncated


def add_flags_from_config(parser, config_dict):
    """
    adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """
    def OrNone(default):
        def func(x):
            # convert "none" to proper None object.
            if x.lower() == "none": return None
            # if default is None (and x is not None), return x without conversion as str.
            elif default is None: return str(x)
            # otherwise, default has non-None type; convert x to that type.
            else: return type(default)(x)
        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument.
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                        help=description
                    )
                else: parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else: parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser


def save_model(encoder, decoder, optimizer, epoch, loss, save_path, logger):
    """
    save model checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss, 'args': None
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(encoder, decoder, optimizer, load_path, device, logger):
    """
    load model checkpoint
    """
    if not os.path.exists(load_path):
        logger.error(f"Model file not found: {load_path}")
        return False
    
    checkpoint = torch.load(load_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Model loaded from {load_path}")
    logger.info(f"Loaded from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    return True


def eval_model(encoder, decoder, datagen, device, logger, bs= 20, p_e=0.1, histo_clen=True):
    encoder.eval()
    decoder.eval()
    
    total_d = 0.0
    total_correct = 0
    total_samples = 0
    total_length = 0.0
    code_lengths = [] if histo_clen else None
    
    # store per-sample metrics for confidence interval calculation
    sample_ds = []
    sample_accs = []
    sample_lengths = []
    
    logger.info(f"Evaluating on test set with p_e={p_e}...")
    
    with torch.no_grad():
        for b, (x, y) in enumerate(datagen.batch(train=False, batch_size=bs)):
            x = x.to(device)
            y = y.to(device)
            x = x.view(x.size(0), -1)

            l, _, zt, _, _, _ = encoder.encode(x)
            z = truncate(zt, l)
            zh = channel(z, p_e=p_e)
            
            yh, _ = decoder.decode(zh)

            # calculate per-sample d
            d_per_sample = F.cross_entropy(yh, y, reduction='none')
            sample_ds.extend(d_per_sample.cpu().numpy().tolist())
            
            # calculate per-sample accuracy
            pred = torch.argmax(yh, dim=1)
            correct_per_sample = (pred == y).float()
            sample_accs.extend(correct_per_sample.cpu().numpy().tolist())
            
            # accumulate totals
            total_d += d_per_sample.sum().item()
            total_correct += correct_per_sample.sum().item()
            total_samples += y.size(0)
            total_length += l.float().mean().item() * y.size(0)
            
            # record individual code lengths for CI calculation
            sample_lengths.extend(l.cpu().tolist())
            
            # record individual code lengths for histogram if requested
            if histo_clen and code_lengths is not None: code_lengths.extend(l.cpu().tolist())
            
            if (b + 1) % 20 == 0: logger.info(f"  Processed {total_samples} samples...")
    
    # calculate means
    avg_d = total_d / total_samples
    avg_acc = total_correct / total_samples
    avg_len = total_length / total_samples
    
    # calculate 95% confidence intervals
    sample_ds_arr = np.array(sample_ds)
    sample_accs_arr = np.array(sample_accs)
    sample_lengths_arr = np.array(sample_lengths)
    
    d_std = np.std(sample_ds_arr, ddof=1)
    acc_std = np.std(sample_accs_arr, ddof=1)
    len_std = np.std(sample_lengths_arr, ddof=1)
    
    n = len(sample_ds_arr)
    d_se = d_std / np.sqrt(n)
    acc_se = acc_std / np.sqrt(n)
    len_se = len_std / np.sqrt(n)
    
    # 95% confidence interval (z-score = 1.96 for large samples, e.g., n > 30)
    z_score = 1.96
    d_ci_low = avg_d - z_score * d_se
    d_ci_up = avg_d + z_score * d_se
    acc_ci_low = avg_acc - z_score * acc_se
    acc_ci_up = avg_acc + z_score * acc_se
    len_ci_low = avg_len - z_score * len_se
    len_ci_up = avg_len + z_score * len_se
    
    logger.info(f"Evaluation completed: {total_samples} samples processed")
    logger.info(f"Average code length (rate): {avg_len:.2f}, 95% CI: [{len_ci_low:.2f}, {len_ci_up:.2f}]")
    logger.info(f"Average distortion: {avg_d:.4f}, 95% CI: [{d_ci_low:.4f}, {d_ci_up:.4f}]")
    logger.info(f"Average accuracy: {avg_acc:.4f}, 95% CI: [{acc_ci_low:.4f}, {acc_ci_up:.4f}]")
    
    # create histogram if requested
    histogram = None
    if histo_clen and code_lengths is not None:
        max_length = max(code_lengths)
        histogram = [0] * (max_length + 1)
        for length in code_lengths:
            histogram[length] += 1
        logger.info(f"Code length histogram created: lengths from 0 to {max_length}")

    print(histogram)
    
    return histogram



if __name__ == "__main__":
    pass
