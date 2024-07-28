import numpy as np
import collections
import numpy as np
import torch
import torchvision.transforms as transforms

word_dict = None
word_list = None
_pad = "<pad>"
_bos = "<bos>"
_eos = "<eos>"
"""
This code follows the steps of preprocessing in tff shakespeare dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/shakespeare_dataset.py
"""

SEQUENCE_LENGTH = 90  # from McMahan et al AISTATS 2017
# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
CHAR_VOCAB = list(
    "dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#'/37;?bfjnrvzBFJNRVZ\"&*.26:\naeimquyAEIMQUY]!%)-159\r"
)


def get_word_dict():
    global word_dict
    if word_dict == None:
        words = [_pad] + CHAR_VOCAB + [_bos] + [_eos]
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict


def get_word_list():
    global word_list
    if word_list == None:
        word_dict = get_word_dict()
        word_list = list(word_dict.keys())
    return word_list


def id_to_word(idx):
    return get_word_list()[idx]


def char_to_id(char):
    word_dict = get_word_dict()
    if char in word_dict:
        return word_dict[char]
    else:
        return len(word_dict)


def preprocess(sentences, max_seq_len=SEQUENCE_LENGTH):

    sequences = []

    def to_ids(sentence, num_oov_buckets=1):
        """
        map list of sentence to list of [idx..] and pad to max_seq_len + 1
        Args:
            num_oov_buckets : The number of out of vocabulary buckets.
            max_seq_len: Integer determining shape of padded batches.
        """
        tokens = [char_to_id(c) for c in sentence]
        tokens = [char_to_id(_bos)] + tokens + [char_to_id(_eos)]
        if len(tokens) % (max_seq_len + 1) != 0:
            pad_length = (-len(tokens)) % (max_seq_len + 1)
            tokens += [char_to_id(_pad)] * pad_length
        return (
            tokens[i : i + max_seq_len + 1]
            for i in range(0, len(tokens), max_seq_len + 1)
        )

    for sen in sentences:
        sequences.extend(to_ids(sen))
    return sequences


def split(dataset):
    ds = np.asarray(dataset)
    x = ds[:, :-1]
    y = ds[:, 1:]
    return x, y


if __name__ == "__main__":
    print(
        split(
            preprocess(
                [
                    "Yonder comes my master, your brother.",
                    "Come not within these doors; within this roof\nThe enemy of all your graces lives.\nYour brother- no, no brother; yet the son-\nYet not the son; I will not call him son\nOf him I was about to call his father-\nHath heard your praises; and this night he means\nTo burn the lodging where you use to lie,\nAnd you within it. If he fail of that,\nHe will have other means to cut you off;\nI overheard him and his practices.\nThis is no place; this house is but a butchery;\nAbhor it, fear it, do not enter it.\nNo matter whither, so you come not here.",
                    "To the last gasp, with truth and loyalty.\nFrom seventeen years till now almost four-score\nHere lived I, but now live here no more.\nAt seventeen years many their fortunes seek,\nBut at fourscore it is too late a week;\nYet fortune cannot recompense me better\nThan to die well and not my master's debtor.          Exeunt\nDear master, I can go no further. O, I die for food! Here lie",
                    "[Coming forward] Sweet masters, be patient; for your father's",
                    "remembrance, be at accord.\nIs 'old dog' my reward? Most true, I have lost my teeth in",
                ]
            )
        )
    )





"""
preprocess reference : https://github.com/google-research/federated/blob/master/utils/datasets/cifar100_dataset.py
"""


# def cifar100_transform(img_mean, img_std, train=True, crop_size=(24, 24)):
def cifar100_transform(img_mean, img_std, train=True, crop_size=32):
    """cropping, flipping, and normalizing."""
    if train:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
                Cutout(16),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )
        # return transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.CenterCrop(crop_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=img_mean, std=img_std),
        #     ]
        # )


def preprocess_cifar_img(img, train):
    # scale img to range [0,1] to fit ToTensor api
    img = torch.div(img, 255.0)
    transoformed_img = torch.stack(
        [
            cifar100_transform(
                [0.5071, 0.4865, 0.4409],
                [0.2673, 0.2564, 0.2762],
                train,
            )(i.permute(2, 0, 1))
            for i in img
        ]
        # [
        #     cifar100_transform(
        #         i.type(torch.DoubleTensor).mean(),
        #         i.type(torch.DoubleTensor).std(),
        #         train,
        #     )(i.permute(2, 0, 1))
        #     for i in img
        # ]
    )
    return transoformed_img



class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


