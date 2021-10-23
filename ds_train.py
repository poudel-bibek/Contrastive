import time 
import random 
import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model import PreModel
from optimizer import LARS
from loss import SimCLR_Loss

from torch.utils.data import Dataset, DataLoader
from data_utils import DataGenerator, DownstreamDataGenerator
from data_utils import load_data
from ds_model import DSModel

class DownstreamTrainer:
    def __init__(self, args):
        self.args = args 
        self.writer = SummaryWriter()

        print("\n--------------------------------")
        print("Seed: ", self.args.seed)

        # Set seeds
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        ## Identify device and acknowledge
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        print("Device Assigned to: ", self.device)

        ## Data Loading operations
        train_images, train_targets, val_images, val_targets  = load_data(self.args.dataset_src)

        # The same dataset that was used to train the Encoder with contrastive learning is now being used to train the classification layer
        # I matched the portion of the dataset that I used for training the SimCLR encoder for the downstream task to keep them similar
        train_images = train_images[:40000]
        train_targets = train_targets[:40000]

        val_images = val_images[:20000]
        val_targets = val_targets[:20000]

        print("Data Directory: ", self.args.dataset_src)
        print("\nLoaded:\nTraining: {} Images, {} Targets\nValidation: {} Images, {} Targets".format(train_images.shape[0],
                                                                                                    train_targets.shape[0],
                                                                                                    val_images.shape[0],
                                                                                                    val_targets.shape[0]))

        # Num classes is set to 1000 since that's how many there are in mini ImageNet
        self.datagen_train = DownstreamDataGenerator('train', train_images, train_targets, num_classes=1001)
        self.train_dataloader = DataLoader(self.datagen_train, self.args.train_batch_size, drop_last = True)

        self.datagen_val = DownstreamDataGenerator('train', val_images, val_targets, num_classes=1001)
        self.val_dataloader = DataLoader(self.datagen_val, self.args.val_batch_size, drop_last = True)

        self.premodel = torch.load('./Saved_models/trained_model.pt')

        self.net = DSModel(self.premodel, 1001).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD([params for params in self.net.parameters() if params.requires_grad], lr=0.01, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.98, last_epoch=-1, verbose=True)

        print("\n--------------------------------")
        print("Total No. of Trainable Parameters: ",sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    def train(self):
        train_loss_collector = np.zeros(self.args.train_epochs)
        train_accuracy_collector = np.zeros(self.args.train_epochs)

        val_loss_collector = np.zeros(self.args.train_epochs) # Validate every epoch as well
        val_accuracy_collector = np.zeros(self.args.train_epochs)

        best_loss = float('inf')
        logfile = open('./logs/ds_logfile.txt', 'w')

        print("\n#### Started Training ####")
        logfile.write("\n#### Started Training ####\n")

        for i in range(self.args.train_epochs):
            self.net.train()

            acc_sublist = np.array([])

            if i==0:
                myfile = open('./logs/ds_model_init_weights.txt', 'w')
                myfile.write("SEED = %s\n" % self.args.seed)
                print("Writing Model Initial Weights to a file\n")
                for param in self.net.parameters():
                    myfile.write("%s\n" % param.data)
                myfile.close()

            start = time.time()
            batch_loss_train = 0 

            print("Ep. {}/{}:".format(i+1, self.args.train_epochs), end="\t")
            logfile.write("Ep. {}/{}:\t".format(i+1, self.args.train_epochs))
            ground_truths_train =[]
            predictions_train =[]

            correct = 0
            total = 0

            for bi, (img, label) in enumerate(self.train_dataloader):
                img = img.squeeze().to(self.device).float()
                label = label.type(torch.LongTensor)
                label = label.to(self.device)

                pred = self.net(img)

                self.optimizer.zero_grad() 
                loss = self.criterion(pred, label)

                loss.backward()

                # Not sure why this line is specifically placed inbetween the loss.backward call and the optimizer.step call or if it needs to be here
                #preds = torch.exp(pred) / torch.sum(torch.exp(pred))
                _, predicted = torch.max(pred.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                self.optimizer.step() 

                loss_np = loss.cpu().detach().clone().numpy()
                self.writer.add_scalar("Batch Loss, Train:", loss_np, bi)

                # I have no confidence if this line is doing what it is supposed to do (as in actually calculate the accuracy)
                #acc_sublist = np.append(acc_sublist, np.array(np.argmax(preds.cpu().detach().clone().numpy(),axis=1)==label.cpu().detach().clone().numpy()).astype('int'),axis=0)


                batch_loss_train += loss_np 


            # Average Batch Loss per epoch
            avg_batch_loss_train = batch_loss_train / len(self.train_dataloader)
            print("Train: ABL {}".format(round(avg_batch_loss_train,3)), end="\t")
            logfile.write("Train: ABL {}".format(round(avg_batch_loss_train,3)))

            # Average Accuracy per epoch
            #avg_accuracy_train = np.mean(acc_sublist)
            avg_accuracy_train = (100 * correct / total)
            print("Train: ACC {}".format(round(avg_accuracy_train,3)), end="\t")
            logfile.write("Train: ACC {}".format(round(avg_accuracy_train,3)))

            # Validating here
            avg_batch_loss_val, avg_accuracy_val = self.validate()

            # Average Validation Batch Loss per epoch
            print("Val: ABL {}".format(round(avg_batch_loss_val,3)), end="\t")
            logfile.write("Val: ABL {}".format(round(avg_batch_loss_val,3)))

            # Average Validation Accuracy per epoch
            print("Val: ACC {}".format(round(avg_accuracy_val,3)), end="\t")
            logfile.write("Val: ACC {}".format(round(avg_accuracy_val,3)))

            print("Time: {} s".format(round(time.time() - start, 1))) #LR: {}".format(round(time.time() - start, 1), self.optimizer.param_groups[0]['lr'] )) 
            logfile.write("Time: {} s".format(round(time.time() - start, 1)))

            # Generally should be looking at validation loss here but..
            if avg_batch_loss_val < best_loss:
                best_loss = avg_batch_loss_val
                print("#### New Model Saved #####")
                logfile.write("#### New Model Saved #####\n")
                torch.save(self.net, './Saved_models/ds_trained_model.pt')
            
            self.lr_scheduler.step()

            self.datagen_train.on_epoch_end()
            self.datagen_val.on_epoch_end()

            train_loss_collector[i] = avg_batch_loss_train
            train_accuracy_collector[i] = avg_accuracy_train

            val_loss_collector[i] = avg_batch_loss_val
            val_accuracy_collector[i] = avg_accuracy_val

        self.writer.flush() 
        self.writer.close()

        # Draw loss plot (both train and val)
        fig, ax = plt.subplots(figsize=(16,5), dpi = 100)
        xticks= np.arange(0,self.args.train_epochs,50)

        ax.set_ylabel("MSE Loss (Training )") # & Validation
        ax.plot(np.asarray(train_loss_collector))
        ax.plot(np.asarray(val_loss_collector))

        ax.set_xticks(xticks) #;
        ax.legend(["Training", "Validation"]) # ["Validation", "Training"]
        fig.savefig('./logs/ds_training_result.png')

        print("#### Ended Training ####")
        logfile.write("#### Ended Training ####")
        logfile.close()
        # Plot AMA as well

    def validate(self):     
        self.net.eval()

        with torch.no_grad():
            acc_sublist = np.array([])
            batch_loss_val = 0
            
            correct = 0
            total = 0

            for bi, (img, label) in enumerate(self.val_dataloader):
                img = img.squeeze().to(self.device).float()
                label = label.type(torch.LongTensor)
                label = label.to(self.device)
        
                pred = self.net(img)

                self.optimizer.zero_grad() 
                loss = self.criterion(pred, label)

                # Not sure why this line is specifically placed inbetween the loss.backward call and the optimizer.step call or if it needs to be here
                #preds = torch.exp(pred) / torch.sum(torch.exp(pred))

                _, predicted = torch.max(pred.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                loss_np = loss.cpu().detach().numpy()
                self.writer.add_scalar("Batch Loss, Val:", loss_np, bi)

                #acc_sublist = np.append(acc_sublist, np.array(np.argmax(preds.cpu().detach().clone().numpy(),axis=1)==label.cpu().detach().clone().numpy()).astype('int'),axis=0)

                batch_loss_val += loss_np

        avg_batch_loss_val =  batch_loss_val / len(self.val_dataloader)
        #avg_accuracy_val = np.mean(acc_sublist)
        avg_accuracy_val = (100 * correct / total)

        return avg_batch_loss_val, avg_accuracy_val


    

def main(args):
    tr = DownstreamTrainer(args)
    tr.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", default = "./Data", help = "Data Directory")
    parser.add_argument("--seed", type = int, default = 99, help = "Randomization Seed")
    parser.add_argument("--train_batch_size",type = int, default = 4, help = "Train batch size")
    parser.add_argument("--temperature",type = int, default = 0.5, help = "For the Loss function")
    parser.add_argument("--val_batch_size",type = int, default = 4, help = "Train batch size")
    parser.add_argument("--train_epochs", type = int, default = 10, help = "Number of epochs to do training")
    parser.add_argument("--lr", type=float, default = 0.01, help = "Learning Rate")
    parser.add_argument("--weight_decay", type=float, default = 1e-6, help = "Weight Decay")

    parser.add_argument("--dataset_src", default='./Data/Imagenet/64/', help="The source for creating dataset")
    main(parser.parse_args())