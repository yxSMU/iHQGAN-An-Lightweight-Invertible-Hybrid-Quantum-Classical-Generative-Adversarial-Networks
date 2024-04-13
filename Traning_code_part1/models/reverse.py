import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cpu')
#第二个类的定义，每一个类别包括一个生成器和判别器，对的
class PQWGAN_CC3():
    def __init__(self, image_size,image_size_2, channels, n_generators, n_qubits, n_ancillas, n_layers, patch_shape):
        self.image_shape = (channels,image_size, image_size_2)
        self.critic = self.ClassicalCritic(self.image_shape)
        self.generator = self.QuantumGenerator(n_generators, n_qubits, n_ancillas, n_layers, self.image_shape, patch_shape)
#判别器
    class ClassicalCritic(nn.Module):
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape

            self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)
#生成器
    class QuantumGenerator(nn.Module):
        def __init__(self, n_generators, n_qubits, n_ancillas, n_layers, image_shape, patch_shape):
            super().__init__()
            self.n_generators = n_generators
            self.n_qubits = n_qubits
            self.n_ancillas = n_ancillas
            self.n_layers = n_layers
            self.q_device = qml.device("default.qubit", wires=n_qubits)
            self.params = nn.ParameterList([nn.Parameter(torch.rand(n_layers, n_qubits, 3), requires_grad=True) for _ in range(n_generators)])
            self.qnode = qml.QNode(self.reversible_variational_circuit, self.q_device, interface="torch",diff_method='backprop')
            self.image_shape = image_shape
            self.patch_shape = patch_shape

        def forward(self, x):

            x = x.view(x.shape[0], -1)

            special_shape = bool(self.patch_shape[0]) and bool(self.patch_shape[1])
            patch_size = 2 ** (self.n_qubits - self.n_ancillas)

            image_pixels = self.image_shape[1] * self.image_shape[2]
            pixels_per_patch = image_pixels // self.n_generators

            if special_shape and self.patch_shape[0] * self.patch_shape[1] != pixels_per_patch:
                raise ValueError("patch shape and patch size dont match!")
            output_images = torch.Tensor(x.size(0), 0)


            i = 0
            for sub_generator_param in self.params:
                patches = torch.Tensor(0, pixels_per_patch)
                start = i * (x.shape[1] // 32)  # 计算每个部分的起始位置
                end = (i + 1) * (x.shape[1] // 32)  # 计算每个部分的结束位置
                i = i + 1
                for item in x[:, start:end]:
                    sub_generator_out = self.partial_trace_and_postprocess(item, sub_generator_param).float().unsqueeze(0)
                    if pixels_per_patch < patch_size:
                        sub_generator_out = sub_generator_out[:, :pixels_per_patch]
                    patches = torch.cat((patches, sub_generator_out))
                output_images = torch.cat((output_images, patches), 1)


            if special_shape:
                final_out = torch.zeros(x.size(0), *self.image_shape)
                for i,img in enumerate(output_images):
                    for patches_done, j in enumerate(range(0, img.shape[0], pixels_per_patch)):
                        patch = torch.reshape(img[j:j+pixels_per_patch], self.patch_shape)
                        starting_h = ((patches_done * self.patch_shape[1]) // self.image_shape[2]) * self.patch_shape[0]
                        starting_w = (patches_done * self.patch_shape[1]) % self.image_shape[2]
                        final_out[i, 0, starting_h:starting_h+self.patch_shape[0], starting_w:starting_w+self.patch_shape[1]] = patch
            else:
                final_out = output_images.view(output_images.shape[0], *self.image_shape)
            return final_out

        # 定义可逆的变分量子电路
        def reversible_variational_circuit(self, inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), pad_with=0., normalize=True)
            for i in reversed(range(self.n_layers)):
                for j in reversed(range(self.n_qubits - 1)):
                    qml.CNOT(wires=[j, j + 1])
                for j in reversed(range(self.n_qubits)):
                    qml.Rot(-weights[i][j][0], -weights[i][j][1], -weights[i][j][2], wires=j)

            return qml.probs(wires=list(range(self.n_qubits)))


        def partial_trace_and_postprocess(self, latent_vector, weights):
            probs = self.qnode(latent_vector, weights)
            probs_given_ancilla_0 = probs[:2 ** (self.n_qubits - self.n_ancillas)]
            post_measurement_probs = probs_given_ancilla_0 / torch.sum(probs_given_ancilla_0)
            post_processed_patch = ((post_measurement_probs / torch.max(post_measurement_probs)) - 0.5) * 2
            return post_processed_patch
