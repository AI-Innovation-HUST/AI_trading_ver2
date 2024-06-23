import torch
import torch.nn as nn
from model import *
from dataloader_v2 import *
from sklearn.model_selection import train_test_split
import ta
from scaler import *
import time
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
os.makedirs("results",exist_ok=True)



def compute_acc(pred_high,pred_low, ground_truth,threshold=0.08):
    diff_h = (torch.abs((pred_high - ground_truth[:,0]) / ground_truth[:,0])).sum()/pred_high.size(0)
    diff_l =(torch.abs((pred_low - ground_truth[:,1]) / ground_truth[:,1])).sum()/pred_low.size(0)

    return diff_h,diff_l



def load_data(csv_file, test_size=0.1, val_size=0.2):
    
    df = pd.read_csv(csv_file)
    df = ta.add_all_ta_features(df, "o", "h", "l", "c", "vol", fillna=True)
    df.dropna(inplace=True)
    scaler = ScalerData(method='minmax') # test method
    
    df = scaler.fit_transform(df)
    save_scaler_file = "results/scaler.pkl"
    pickle.dump(scaler, open(save_scaler_file, 'wb'))
    print("Save scaler model successfully")
    train_val_df, test_df = train_test_split(df, test_size=test_size, shuffle=False, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, shuffle=False, random_state=42)

    train_dataset = CoinDataset(train_df)
    val_dataset = CoinDataset(val_df.reset_index(drop=True))
    test_dataset = CoinDataset(test_df.reset_index(drop=True))

    return train_dataset, val_dataset, test_dataset

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        return self.early_stop

def plot_diagram(e,data1,data2,label1,label2,label3):
    plt.plot(e, data1, label=label1)
    plt.plot(e, data2, label=label2)
    plt.title('training transformer')
    plt.xlabel('Epoch')
    plt.ylabel(label3)
    plt.legend()
    plt.grid(True)
    plt.savefig("{}_{}.png".format(label1,label2))
    plt.clf()

if __name__ == '__main__':
    data = []
    dev = torch.device("cpu")
    train_dataset, val_dataset, test_dataset = load_data(csv_file='Dataset/data_1714496400000_1717174800000.csv')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    model = Transformer(n_blocks=4,d_model=16,n_heads=8,d_ff=256,dropout=0.5)
    model.to(dev)
    
    criterion_high = nn.HuberLoss(delta=0.5)
    criterion_low = nn.HuberLoss(delta=0.5)
    lr = 0.0001 # learning rate
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.003)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    accuracies = []
    lossies = []
    val_loss = []
    epochs = 2
    train_loss_high = []
    train_loss_low = []
    valid = []
    check_acc_h_train = []
    check_acc_l_train = []
    val_acc_h = []
    val_acc_l = []
    for epoch in range(epochs):
        count = 0
        cum_loss_h = 0
        cum_loss_l = 0
        acc_count = 0
        accs_h = 0
        accs_l = 0
        train_l_high = []
        train_l_low = []
        val_l = []
        for batch in tqdm(train_loader):
            data, targets = batch
            targets = targets.view(-1,2).to(dev)
            high,low = model(data.to(dev))
            high = high.view(-1,1)
            low = low.view(-1,1)
            # print(output.shape)
            loss_high = criterion_high(high.float(),targets[:,0].float())
            loss_low = criterion_low(low.float(),targets[:,1].float)
            total_loss = 0.5 * loss_high + 0.5 * loss_low
            train_l_high.append(loss_high.item())
            train_l_low.append(loss_low.item())
            score_h,score_l = compute_acc(high,low,targets)
            accs_h += score_h
            accs_l += score_l
            cum_loss_h += loss_high
            cum_loss_l += loss_low
            total_loss.backward()
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            count+=1
        train_ls_h = (cum_loss_h/count).item()
        train_ls_l = (cum_loss_l/count).item()

        train_loss_high.append(train_ls_h)
        train_loss_low.append(train_loss_low)
        check_acc_h_train.append(float((accs_h/count).item()))
        check_acc_l_train.append(float((accs_l/count).item()))
        
        
        print(epoch,"Loss high: ",(cum_loss_h/count).item(),"Loss_low:",(cum_loss_l/count).item(),"Total loss: ",(total_loss/count).item(),"ACC high train:",(accs_h/count).item(),"ACC low train:",(accs_l/count).item())
        print("------START EVAL------")
        eval_loss,acc_h,acc_l = evaluate(model,epoch,criterion_high,criterion_low,val_loader,dev=dev)
        valid.append(eval_loss)
        val_acc_h.append(acc_h)
        val_acc_l.append(acc_l)
        print(epoch,"Loss: ",(cum_loss_h/count).item()," Valid_loss: ",eval_loss,"Valid high acc: ",acc_h,"valid low acc",acc_l)
        if len(val_loss)>0 and eval_loss < val_loss[-1]:
            val_loss.append(eval_loss)
            torch.save(model,"results/evalModel_best.pth")
        else:
            val_loss.append(eval_loss)
            torch.save(model,"results/evalModel_best1.pth")
        # early_stop = early_stopping(train_ls,eval_loss)
        # if early_stop:
        #   torch.save(model,'best_model_{}.pth'.format(epoch+1))
        #   print(" training done at {} epochs".format(epoch+1))
        #   break
    
    # print(valid)
    # e = [i for i in range(epochs)]
    # plot_diagram(e,train_loss,valid,"Train loss","Valid loss","Loss")
    # plot_diagram(e,check_acc_h_train,val_acc_h,"Train high accuracy","Valid high accuracy","Accuracy high")
    # plot_diagram(e,check_acc_l_train,val_acc_l,"Train low accuracy","Valid low accuracy","Accuracy low")





      