import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocess import sample_rate
from model import Generator, Discriminator
from utils_main import AudioDataset, emphasis
from utils_wgan import gradient_penalty

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=86, type=int, help='train epochs number')

    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    # load data
    print('loading data...')
    train_dataset = AudioDataset(data_type='train')
    test_dataset = AudioDataset(data_type='test')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # generate reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE)

    # create D and G instances
    discriminator = Discriminator()
    generator = Generator()
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)
    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))
    # optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)

    if not os.path.exists("../save1/results"):
        os.makedirs("../save1/results")
    if not os.path.exists("../save1/epochs"):
        os.makedirs("../save1/epochs")
    log_file = open("log1.txt",'w')

    for epoch in range(NUM_EPOCHS):
        # train_bar = tqdm(train_data_loader)
        step = 0
        for train_batch, train_clean, train_noisy in train_data_loader:
            step+=1
            # latent vector - normal distribution
            z = nn.init.normal_(torch.Tensor(train_batch.size(0), 1024, 8))
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # TRAIN D to recognize clean audio as clean
            # training batch pass
            generated_outputs = generator(train_noisy, z)
            train_batch_noise_ = torch.cat((generated_outputs, train_noisy), dim=1)
            train_batch_noise = Variable(train_batch_noise_)
            train_batch_clean_ = torch.cat((train_clean, train_noisy), dim=1)
            train_batch_clean = Variable(train_batch_clean_)
            
            
            for k in range(5):
                discriminator.zero_grad()
                outputs_clean = discriminator(train_batch, ref_batch)
                outputs_noise = discriminator(train_batch_noise, ref_batch)
                gp = gradient_penalty(discriminator,train_batch,train_batch_noise,ref_batch)
                d_loss = outputs_clean.mean() - outputs_noise.mean() + 20*gp
                d_loss.backward()
                print("inner_step:",k,"d_loss:",d_loss.item())
                d_optimizer.step()

            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(train_noisy, z)
            train_batch_noise_ = torch.cat((generated_outputs, train_noisy), dim=1)
            outputs_noise = discriminator(train_batch_noise_, ref_batch)


            
            g_loss_ = 0.5*outputs_noise.mean()
            #g_loss_ = 0.0 * outputs_noise.mean()

            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss

            # backprop + optimize
            g_loss.backward()
            g_optimizer.step()
            log_str = 'Epoch {}({}/{}): d_clean_loss {:.4f}, d_noisy_loss {:.4f}, g_loss {:.4f}, g_conditional_loss {:.4f},' \
                      'gp:{:.4f}'.format(epoch + 1,step,len(train_data_loader),
                                         outputs_clean.mean().item(), outputs_noise.mean().item(),
                                         g_loss.item(), g_cond_loss.item(),gp.item())
            log_file.write(log_str+'\n')
            log_file.flush()
            print(log_str)

        # TEST model
        test_bar = tqdm(test_data_loader, desc='Test model and save generated audios')
        for test_file_names, test_noisy in test_bar:
            z = nn.init.normal(torch.Tensor(test_noisy.size(0), 1024, 8))
            if torch.cuda.is_available():
                test_noisy, z = test_noisy.cuda(), z.cuda()
            test_noisy, z = Variable(test_noisy), Variable(z)
            fake_speech = generator(test_noisy, z).data.cpu().numpy()  # convert to numpy array
            fake_speech = emphasis(fake_speech, emph_coeff=0.95, pre=False)

            for idx in range(fake_speech.shape[0]):
                generated_sample = fake_speech[idx]
                file_name = os.path.join('../save1/results',
                                         '{}_e{}.wav'.format(test_file_names[idx].replace('.npy', ''), epoch + 1))
                wavfile.write(file_name, sample_rate, generated_sample.T)

        # save the model parameters for each epoch
        g_path = os.path.join('../save1/epochs', 'generator-{}.pkl'.format(epoch + 1))
        d_path = os.path.join('../save1/epochs', 'discriminator-{}.pkl'.format(epoch + 1))
        torch.save(generator.state_dict(), g_path)
        torch.save(discriminator.state_dict(), d_path)
