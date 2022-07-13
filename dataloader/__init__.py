from torch.utils.data import DataLoader
from .outdoor360 import Outdoor360
from .maverick_dataloader import Maverick, MaverickRaw


def data_loader(args):
    test_set = Outdoor360(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return test_loader


def maverick_raw_loader(args):
    test_set = MaverickRaw(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return test_loader


def maverick_loader(args):
    test_set = Maverick(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return test_loader
