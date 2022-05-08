import torch

train_dict = torch.load('train.pt')
val_dict = torch.load('val.pt')
test_dict = torch.load('test.pt')

window_len = 206
train_dict['samples'] = train_dict['samples'][:window_len,:,:]
val_dict['samples'] = val_dict['samples'][:window_len,:,:]
test_dict['samples'] = test_dict['samples'][:window_len,:,:]

torch.save(train_dict,'train_cropped.pt')
torch.save(val_dict,'val_cropped.pt')
torch.save(test_dict,'test_cropped.pt')
