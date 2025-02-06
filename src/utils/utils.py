import torch
import torchaudio

def train(model, train_loader, loss_function, optimizer, epochs, device, save_path=None):
    loss_curve = list()
    min_loss_curve = float('INF')

    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        average_loss = overall_loss/((batch_idx+1)*train_loader.batch_size)
        loss_curve.append(average_loss)

        print("\t[TRAINING] Epoch", epoch + 1, "\tAverage Loss: ", average_loss)

        if save_path and len(loss_curve) > 0 and average_loss < min_loss_curve:
            torch.save(model.state_dict(), save_path)
            print(f'\t[INFO] Best model saved at epoch {epoch + 1}, loss: {average_loss}')
            min_loss_curve = average_loss
        
    return loss_curve


def save_sample(sample, path):
    b = torch.cat([sample, sample], dim=0)
    b = b.detach().to('cpu')
    torchaudio.save(path, b[:2], 3000)
    return b

