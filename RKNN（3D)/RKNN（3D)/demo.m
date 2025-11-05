clear
clc
model_gen= gen_model_gen;
% load truth_target.mat
truth_target = gen_target(model_gen);
handles= plot_truth(truth_target);