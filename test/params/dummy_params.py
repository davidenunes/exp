from exp.params import ParamSpace
import os

default_out_dir = os.getcwd()
home = os.getenv("HOME")
default_corpus = os.path.join(home, "data/datasets/ptb/")

ps = ParamSpace("baseline.params")
# prefix used to identify result files

# number of hidden layers, hidden layer dimensions and activations
ps.add_list("a", [1, 2])
ps.add_range("b", 2)



print("param size: ", ps.grid_size)
ps.write("test.params")

for param in ps.param_grid(include_id=True, nruns=True, runs=2):
    print(param)
