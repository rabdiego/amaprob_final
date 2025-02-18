import torch
import torchaudio

def train(model, train_loader, test_loader, loss_function, optimizer, epochs, device, save_path=None):
    train_loss_curve = list()
    test_loss_curve = list()
    
    min_loss_curve = float('INF')
    best_epoch = 0

    model.train()
    for epoch in range(epochs):
        overall_train_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        average_train_loss = overall_train_loss/((batch_idx+1)*train_loader.batch_size)
        train_loss_curve.append(average_train_loss)

        overall_test_loss = 0
        with torch.no_grad():
            for batch_idx, x in enumerate(test_loader):
                x = x.to(device)

                x_hat, mean, log_var = model(x)
                loss = loss_function(x, x_hat, mean, log_var)
                
                overall_test_loss += loss.item()

        average_test_loss =  overall_test_loss/((batch_idx+1)*test_loader.batch_size)
        test_loss_curve.append(average_test_loss)

        print("\t[TRAINING] Epoch", epoch + 1, "\tAverage train Loss: ", average_train_loss, "\tAverage test Loss: ", average_test_loss)

        if save_path and len(test_loss_curve) > 0 and average_test_loss < min_loss_curve:
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f'\t[INFO] Best model saved at epoch {epoch + 1}, test loss: {average_test_loss}')
            min_loss_curve = average_test_loss
        
        if epoch - best_epoch > 30:
            print('\t[INFO] Early stopping')
            break
        
    return train_loss_curve, test_loss_curve


def save_sample(sample, path):
    b = torch.cat([sample, sample], dim=0)
    b = b.detach().to('cpu')
    torchaudio.save(path, b[:2], 3000)
    return b