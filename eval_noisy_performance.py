import librosa
import numpy as np
from utils_eval import *
import glob
import timeit
import argparse
from scipy.io import wavfile,savemat,loadmat

# eval expanded noisy testset with composite metrics
#NOISY_TEST_PATH = 'data/expanded_segan1_additive/noisy_testset'

def main(opts):
    NOISY_TEST_PATH = opts.test_wavs
    CLEAN_TEST_PATH = opts.clean_wavs

    noisy_wavs = glob.glob(os.path.join(NOISY_TEST_PATH, '*.wav'))
    metrics = {'csig':[], 'cbak':[], 'covl':[],'pesq':[],'ssnr':[]}
    timings = []
    #out_log = open('eval_noisy.log', 'w')
    out_log = open(opts.logfile, 'w')
    out_log.write('FILE CSIG CBAK COVL PESQ SSNR\n')
    for n_i in range(len(noisy_wavs)):
        if n_i == 693:
            continue
        noisy_wav = noisy_wavs[n_i]
    # for n_i, noisy_wav in enumerate(noisy_wavs, start=1):
    #     if n_i == 692:
    #         continue
        bname = os.path.splitext(os.path.basename(noisy_wav))[0]
        clean_wav = os.path.join(CLEAN_TEST_PATH, bname + '.wav')
        noisy, rate = librosa.load(noisy_wav, 16000)
        clean, rate = librosa.load(clean_wav, 16000)
        #rate, noisy = wavfile.read(noisy_wav)
        #rate, clean = wavfile.read(clean_wav)
        beg_t = timeit.default_timer()
        csig, cbak, covl, pesq, ssnr = CompositeEval(clean, noisy, True)
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        metrics['csig'].append(csig)
        metrics['cbak'].append(cbak)
        metrics['covl'].append(covl)
        metrics['pesq'].append(pesq)
        metrics['ssnr'].append(ssnr)
        out_log.write('{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3}\n'.format(bname + '.wav',
                                                                      csig,
                                                                      cbak,
                                                                      covl,
                                                                      pesq,
                                                                      ssnr))
        print('Processed {}/{} wav, CSIG:{:.3f} CBAK:{:.3f} COVL:{:.3f} '
              'PESQ:{:.3f} SSNR:{:.3f} '
              'total time: {:.2f} seconds, mproc: {:.2f}'
              ' seconds'.format(n_i, len(noisy_wavs), csig, cbak, covl,
                                pesq, ssnr,
                                np.sum(timings),
                                np.mean(timings)))
    out_log.close()

    print('mean Csig: ', np.nanmean(metrics['csig']))
    print('mean Cbak: ', np.nanmean(metrics['cbak']))
    print('mean Covl: ', np.nanmean(metrics['covl']))
    print('mean pesq: ', np.nanmean(metrics['pesq']))
    print('mean ssnr: ', np.nanmean(metrics['ssnr']))
    savemat("noisy_metrics.mat",metrics)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_wavs', type=str, default="WGAN1/enhanced")
    parser.add_argument('--clean_wavs', type=str, default="WGAN1/clean")
    parser.add_argument('--logfile', type=str, default="log")

    opts = parser.parse_args()

    assert opts.test_wavs is not None
    assert opts.clean_wavs is not None
    assert opts.logfile is not None

    main(opts)
    # metrics = loadmat("noisy_metrics.mat")
    # print('mean Csig: ', np.nanmean(metrics['csig']))
    # print('mean Cbak: ', np.nanmean(metrics['cbak']))
    # print('mean Covl: ', np.nanmean(metrics['covl']))
    # print('mean pesq: ', np.nanmean(metrics['pesq']))
    # print('mean ssnr: ', np.nanmean(metrics['ssnr']))

