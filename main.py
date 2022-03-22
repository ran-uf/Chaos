import data
import torch
import matplotlib.pyplot as plt
import numpy as np

from LSTMCells import Model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # x = data.mackey_glass(length=20000, sample=1)
    # x = data.henon(length=20000)
    x = data.ikeda(length=20000)
    x = np.reshape(x[:, 1], (-1, 10))

    train_data = torch.from_numpy(x[x.shape[0] // 2:][:, :, np.newaxis].astype(np.float32))
    test_data = torch.from_numpy(x[:x.shape[0] // 2][:, :, np.newaxis].astype(np.float32))

    model = Model(input_size=1, hidden_size=32, num_layers=1, output_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    batch_size = 32
    for i in range(1000):
        # avg_loss = []
        for n in range(0, train_data.shape[0], batch_size):
            x = train_data[n: n + batch_size, :-1, :]
            y = train_data[n: n + batch_size, 1:, :]

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = []
        with torch.no_grad():
            for n in range(0, test_data.shape[0], batch_size):
                x = test_data[n: n + batch_size, :-1, :]
                y = test_data[n: n + batch_size, 1:, :]

                pred = model(x)
                loss = criterion(pred, y)
                avg_loss.append(loss.item())

            print('test_loss: ', np.mean(avg_loss))

    torch.save(model.state_dict(), 'ikeda.pkl')
