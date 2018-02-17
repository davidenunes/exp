from exp.params import ParamSpace
import os

default_out_dir = os.getcwd()
home = os.getenv("HOME")
default_corpus = os.path.join(home, "data/datasets/ptb/")

ps = ParamSpace("baseline.params")
# prefix used to identify result files

# data
ps.add_value("corpus", default_corpus)
ps.add_value("ngram_size", 4)
ps.add_value("out_dir", default_out_dir)

# architecture
ps.add_list("embed_dim", [128, 256])
ps.add_value("embed_init", "uniform")
ps.add_value("embed_limits", 0.01)
ps.add_value("logit_init", "uniform")
ps.add_value("logit_limits", 0.01)


# number of hidden layers, hidden layer dimensions and activations
ps.add_list("num_h", [1, 2, 3])

print("param size: ",ps.grid_size)
ps.write("test.params")

