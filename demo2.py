from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import pickle

_tokenizer = _Tokenizer()

def load_clip_to_cpu():
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    model = torch.jit.load(model_path, map_location="cuda").eval()
    print(f"Model loaded from path: {model_path}")
    #Model loaded from path: /home/bowei/.cache/clip/ViT-B-16.pt
    return model




class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))

        prompt = clip.tokenize(ctx_init).cuda()
        print(f"Tokenized ctx_init shape: {prompt.shape}")  # [1,77]
        print(f"Prompt: {prompt}")  #6个非0
        # print(f"Prompts device: {prompt.device}")
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt)
        print(f"Embedding shape: {embedding.shape}")  # [1, 77, 512]

        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        print(f"Context vectors shape: {ctx_vectors.shape}")  # [4, 512]
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        print(f"Initial context Parameter shape: {self.ctx.shape}")  # [4, 512]

        prompt_prefix = ctx_init
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        print(f"Name lengths: {name_lens}")

        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(f"Prompts: {prompts}") 

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        print(f"Tokenized prompts shape: {tokenized_prompts.shape}")  # [10, 77]

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts) 
        print(f"Embedding of tokenized prompts shape: {embedding.shape}")  # [10, 77, 512]

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        print(f"Token prefix shape: {self.token_prefix.shape}")  # [10, 1, 512]
        
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        print(f"Token suffix shape: {self.token_suffix.shape}")  # [10, 72, 512]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "middle"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        print(f"Context shape after unsqueeze and expand: {ctx.shape}")  # [10, 4, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        print(f"Prefix shape: {prefix.shape}")  # [10, 1, 512]
        print(f"Suffix shape: {suffix.shape}")  # [10, 72, 512]

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            print(f"Prompts (end position) shape: {prompts.shape}")

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                print(f"Processing class {i} with name_len: {name_len}")
                prefix_i = prefix[i : i + 1, :, :]
                print(f"prefix_i shape: {prefix_i.shape}")  # [1, 1, 512]
                class_i = suffix[i : i + 1, :name_len, :]
                print(f"class_i shape: {class_i.shape}")  # [1, name_len, 512]
                suffix_i = suffix[i : i + 1, name_len:, :]
                print(f"suffix_i shape: {suffix_i.shape}")  # [1, 72-name_len, 512]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                print(f"ctx_i_half1 shape: {ctx_i_half1.shape}")  # [1, n_ctx//2, 512]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                print(f"ctx_i_half2 shape: {ctx_i_half2.shape}")  # [1, n_ctx//2, 512]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
                print(f"Final prompt shape for class {i}: {prompt.shape}")  # [1, total_length, 512]
            prompts = torch.cat(prompts, dim=0)
            print(f"Prompts (middle position) shape: {prompts.shape}") #[10,77,512]

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
            print(f"Prompts (front position) shape: {prompts.shape}")

        else:
            raise ValueError

        return prompts
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        print(f"Input prompts shape: {prompts.shape}") #[10,77,512]
        print(f"Input tokenized_prompts shape: {tokenized_prompts.shape}") #[10,77]
        
        x = prompts + self.positional_embedding
        print(f"After adding positional_embedding, x shape: {x.shape}") #[10,77,512]
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        print(f"After permuting x, shape: {x.shape}") #[77,10,512]
        
        x = self.transformer(x)
        print(f"After transformer, x shape: {x.shape}") #[[77,10,512]]
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        print(f"After permuting back, x shape: {x.shape}") #[10,77,512]
        
        x = self.ln_final(x) 
        print(f"After layer normalization, x shape: {x.shape}") #[10,77,512]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        print("self.text_projection.shape",self.text_projection.shape) #[512,512]
        print("tokenized_prompts",tokenized_prompts)
        print("x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].shape",x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].shape) #[10,512]
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        print(f"After extracting EOT token features, x shape: {x.shape}")  #[10,512]

        return x



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale


    def forward(self, image_features):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        print("logit_scale",logit_scale)
        print(logit_scale.dtype) #float32
        print(image_features.dtype) #float 32
        print(text_features.dtype) #float 16
        logits = logit_scale * image_features @ text_features.t()
        print("logits.shape",logits.shape)
        return logits


# class Config:
#     class TRAINER:
#         class COOP:
#             N_CTX = 5
#             CTX_INIT = "a photo of a"  # 初始化提示文本

# cfg = Config()
# # 创建 PromptLearner 实例
# prompt_learner = PromptLearner(cfg, classnames, clip_model).cuda()

# # 获取生成的 tokenized prompts
# tokenized_prompts = prompt_learner.tokenized_prompts.cuda()
# print(f"Tokenized prompts: {tokenized_prompts.shape}")  #[10,77]

# # 获取生成的 prompt
# prompts = prompt_learner().cuda() #[10,77,512]


# # 使用 TextEncoder 获取 text features
# text_encoder = TextEncoder(clip_model).cuda()
# text_features = text_encoder(prompts, tokenized_prompts)
# print(f"Text features shape: {text_features.shape}")  # [10, 512]

classnames=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class CustomDataset(Dataset):
    def __init__(self, pkl_path):

        with open(pkl_path, 'rb') as f:
           self.data_dict = pickle.load(f)   # 包含 {label: (n, 512) numpy arrays} 的字典

        # 将数据展平成 [(tensor, label)] 的形式
        self.samples = []
        for label, embeddings in self.data_dict.items():
            print(embeddings.shape)
            tensors = torch.tensor(embeddings)
            self.samples.extend([(tensors[i], label) for i in range(tensors.shape[0])])

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        embedding, label = self.samples[idx]
        return embedding, label


train_dataset = CustomDataset('/home/bowei/Desktop/PyProjs/lbw1/DATA/ciafar10/cifar10_r200_train_features.pkl')
test_dataset = CustomDataset('/home/bowei/Desktop/PyProjs/lbw1/DATA/ciafar10/cifar10_test_features.pkl')


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class Config:
    class TRAINER:
        class COOP:
            N_CTX = 5
            CTX_INIT = "a photo of a"  # 初始化提示文本

cfg = Config()


classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


clip_model = load_clip_to_cpu()

print("Building custom CLIP")
model = CustomCLIP(cfg, classnames, clip_model).cuda()

# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")

print("-------------------------------------------------------------------------------------")
for name, param in model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)
# # 假设 model 是你的网络
# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")

criterion = nn.CrossEntropyLoss()
learning_rate=0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs) #(bs,100)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy
def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy
test_accuracies = []


# # 定义学习率调度器
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)



# 测试每个类别的准确率
def test_per_class(model, dataloader, device):
    class_correct_best = [0] * 10
    class_total_best = [0] * 10
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct_best[label] += (predicted[i] == labels[i]).item()
                class_total_best[label] += 1

    global class_accuracies_best
    class_accuracies_best = [class_correct_best[i] / class_total_best[i]  for i in range(10)]
    for i in range(10):
        print(f'Accuracy of class {i}: {class_accuracies_best[i]:.4f}')


best_test_accuracy = 0.0
best_epoch = 0

num_epochs=100 

# 训练和测试
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    test_accuracy = test(model, test_loader, device)
    # scheduler.step()  # 更新学习率

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # 记录最佳测试准确率及其对应的epoch
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_epoch = epoch + 1
        # 保存最佳epoch对应的模型权重
        torch.save(model.state_dict(), 'best_model.pth')

# # 加载最佳epoch对应的模型权重
# best_model = MyNet().to(device)
# best_model.load_state_dict(torch.load('best_model.pth'))


# test_per_class(best_model, test_dataloader, device)


# with open(r'D:\PycharmProjects\myb1\BigExperimentClassAcc_pkl\cifar10_r200+CLIP-B-16+mlp.pkl', 'wb') as f:
#     pickle.dump(class_accuracies_best, f)

# print(f"Best Test Accuracy: {best_test_accuracy:.4f} at Epoch {best_epoch}")
# print("Testing complete.")
# train(model,train_loader,criterion, optimizer, device)


  

