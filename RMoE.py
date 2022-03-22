import matplotlib.pyplot as plt
import torch
from LSTMCells import LSTMCells


class Experts(torch.nn.Module):
    def __init__(self, models):
        super(Experts, self).__init__()
        self.models = models

        self.num_cells = len(models)
        self.num_trajectories = self.num_cells * self.models[0].hidden_size

    # def rand_state(self):
    #     idx = np.random.randint(0, self.num_cells)
    #     idx_0 = np.random.randint(0, self.trajectories[idx].w.shape[0])
    #     return self.trajectories[idx].w[idx_0]

    # def similarity(self, s):
    #     return self.trajectories(s)

    def forward(self, x, state):
        new_state = []
        output = []
        for idx, m in enumerate(self.models):
            o, _state = m(x, state)
            new_state.append(_state)
            # output.append(torch.sum(self.similarity(_state, idx), dim=1, keepdim=True))
            # output.append(self.similarity(_state))
            output.append(o)

        h = torch.cat([s[0] for s in new_state], dim=1)
        c = torch.cat([s[0] for s in new_state], dim=1)
        # output = self.similarity(new_state)
        output = torch.cat(output, dim=1)
        return output.view(x.shape[0], -1), (h, c)


class MMoE(torch.nn.Module):
    def __init__(self, experts):
        super(MMoE, self).__init__()
        self.experts = experts
        self.memory_size = 20

        mstep = torch.zeros(self.memory_size, self.memory_size)
        for i in range(1, self.memory_size):
            mstep[i, i - 1] = 1

        self.register_parameter('mstep', torch.nn.Parameter(mstep, requires_grad=False))

        self.memory_encode = torch.nn.Sequential(
            torch.nn.Linear(self.memory_size * self.experts.num_trajectories + self.memory_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
        )

        self.gate_linear = torch.nn.Sequential(
            torch.nn.Linear(16, self.experts.num_cells),
            torch.nn.Softmax(dim=1)
        )

        self.switch = torch.nn.Sequential(
            torch.nn.Linear(self.memory_size, 1),
            torch.nn.Sigmoid(),
        )

        self.agent = LSTMCells(16, 16, 2)

        self.register_buffer('memory', None, persistent=False)

    def register_memory(self, batch_size):
        mm = torch.zeros(batch_size, self.memory_size, self.experts.num_trajectories + 1)
        mm[:, :, -1] = 0.5
        self.register_buffer('memory', mm, False)

    def memory_step(self, w):
        self.memory = torch.matmul(self.mstep, self.memory)
        self.memory[:, 0, :] = w

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        self.gate_trajectories = []
        self.similarities = []
        self.errors = []
        self.switches = []
        seq_len = x.shape[1]
        state = None
        pred = torch.rand(x.shape[0], n_features).to(next(self.parameters()).device)
        gate_state = torch.softmax(torch.rand(x.shape[0], self.experts.num_cells), dim=1).to(next(self.parameters()).device)
        os = []
        # p = []
        self.register_memory(batch_size)
        agent_state = None
        for i in range(seq_len):
            error = x[:, i, :] - pred
            self.errors.append(error)
            o, new_state = self.experts(x[:, i], state)

            # self.similarities.append(encoded.data)
            inp = torch.cat([new_state[0].view(batch_size, -1), error], dim=1)
            self.memory_step(inp)
            agent_input = self.memory_encode(self.memory.view(x.shape[0], -1))

            agent_output, agent_state = self.agent(agent_input, agent_state)
            gate = self.gate_linear(agent_output)
            # theta = self.switch(self.memory[:, :, -1])
            theta = torch.mean(torch.abs(self.memory[:, :10, -1]), dim=1) * 2.5
            theta = torch.relu(theta) - torch.relu(theta - 1)
            theta = theta.unsqueeze(1)
            self.switches.append(theta.data.cpu())
            gate = gate * theta + gate_state * (1 - theta)
            self.gate_trajectories.append(gate.data)

            h = torch.bmm(gate.unsqueeze(1), new_state[0])
            c = torch.bmm(gate.unsqueeze(1), new_state[1])
            state = (h, c)
            gate_state = gate
            pred = torch.bmm(gate.unsqueeze(1), o.unsqueeze(2))[:, 0, :]

            os.append(pred)

        return torch.stack(os, dim=1)


if __name__ == "__main__":
    import numpy as np
    import data
    from LSTMCells import MyLSTM

    henon = data.henon(length=100000)
    mackey_glass = data.mackey_glass(length=100000, sample=1)
    ikeda = data.ikeda(length=100000)

    henon = np.reshape(henon[:, 0], (-1, 40))
    mackey_glass = np.reshape(mackey_glass[:], (-1, 40))
    ikeda = np.reshape(ikeda[:, 1], (-1, 40))

    train_honon = henon[:henon.shape[0] // 2]
    test_honon = henon[henon.shape[0] // 2:]
    train_mackey_glass = mackey_glass[:mackey_glass.shape[0] // 2]
    test_mackey_glass = mackey_glass[mackey_glass.shape[0] // 2:]
    train_ikeda = ikeda[:ikeda.shape[0] // 2]
    test_ikeda = ikeda[ikeda.shape[0] // 2:]

    visualize_data = np.hstack([test_honon[0], test_mackey_glass[0], test_ikeda[0],
                                test_mackey_glass[1], test_honon[1], test_ikeda[1]
                                ])

    train_data = []
    for i in range(train_honon.shape[0]):
        # train_data.append(np.hstack([train_honon[i], train_mackey_glass[i]]))
        # train_data.append(np.hstack([train_mackey_glass[i+1], train_honon[i+1]]))
        train_data.append(np.hstack([train_honon[i], train_mackey_glass[i]]))
        train_data.append(np.hstack([train_mackey_glass[i], train_honon[i]]))
        train_data.append(np.hstack([train_mackey_glass[i], train_ikeda[i]]))
        train_data.append(np.hstack([train_ikeda[i], train_mackey_glass[i]]))
        train_data.append(np.hstack([train_honon[i], train_ikeda[i]]))
        train_data.append(np.hstack([train_ikeda[i], train_honon[i]]))
    train_data = np.vstack(train_data)

    test_data = []
    for i in range(test_honon.shape[0]):
        # train_data.append(np.hstack([train_honon[i], train_mackey_glass[i]]))
        # train_data.append(np.hstack([train_mackey_glass[i+1], train_honon[i+1]]))
        test_data.append(np.hstack([test_honon[i], test_mackey_glass[i]]))
        test_data.append(np.hstack([test_mackey_glass[i], test_honon[i]]))
        test_data.append(np.hstack([test_mackey_glass[i], test_ikeda[i]]))
        test_data.append(np.hstack([test_ikeda[i], test_mackey_glass[i]]))
        test_data.append(np.hstack([test_honon[i], test_ikeda[i]]))
        test_data.append(np.hstack([test_ikeda[i], test_honon[i]]))
    # for i in range(0, test_honon.shape[0], 2):
    #     test_data.append(np.hstack([test_honon[i], test_mackey_glass[i]]))
    #     test_data.append(np.hstack([test_mackey_glass[i+1], test_honon[i+1]]))
    test_data = np.vstack(test_data)

    train_data = torch.from_numpy(train_data[:, :, np.newaxis].astype(np.float32))
    test_data = torch.from_numpy(test_data[:, :, np.newaxis].astype(np.float32))
    visualize_data = torch.from_numpy(visualize_data[np.newaxis, :, np.newaxis].astype(np.float32))

    model_honon = MyLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    # model_honon.load_state_dict(torch.load('honon.pkl'))
    model_mackey_glass = MyLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    # model_mackey_glass.load_state_dict(torch.load('mackey_glass.pkl'))
    model_ikeda = MyLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    # model_ikeda.load_state_dict(torch.load('ikeda.pkl'))

    models = torch.nn.ModuleList([model_honon, model_mackey_glass, model_ikeda])

    experts = Experts(models)
    model = MMoE(experts)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=3e-4)
    criterion = torch.nn.MSELoss()

    visualize = True
    batch_size = 32
    for epoch in range(500):
        for i in range(0, train_data.shape[0], batch_size):
            x = train_data[i:i + batch_size, :-1, :]
            y = train_data[i:i + batch_size, 1:, :]

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            avg_loss = []
            for i in range(0, test_data.shape[0], batch_size):
                x = test_data[i:i + batch_size, :-1, :]
                y = test_data[i:i + batch_size, 1:, :]

                pred = model(x)
                loss = criterion(pred, y)
                avg_loss.append(loss.item())
        print('epoch:', epoch, 'test_loss:', np.mean(avg_loss))

        if epoch % 1 == 0:
            model(visualize_data[:, :-1, :])
            errors = torch.cat(model.errors).detach().data.cpu().numpy()
            gates = torch.cat(model.gate_trajectories, dim=0).detach().data.cpu().numpy()

            fig, ax = plt.subplots(ncols=1, nrows=2)
            # fig.suptitle('epoch %d' % (epoch+1))

            plt.subplot(211)
            plt.title('predict error')
            plt.plot(errors)
            plt.subplot(212)

            plt.title('gates')
            plt.plot(gates)
            plt.ylim([0, 1.2])
            plt.legend(['honon', 'mackey_glass', 'ikeda'])
            fig.tight_layout()
            plt.show()

    # torch.save(model.state_dict(), 'RMoE.pkl')
