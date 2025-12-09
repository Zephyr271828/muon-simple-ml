# muon-simple-ml

## Paper \& Blogs
### Muon
- [Old Optimizer, New Norm: Anthology](https://arxiv.org/abs/2409.20325)
- [Deriving Muon](https://jeremybernste.in/writing/deriving-muon)
- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
- [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568)
- [Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265)

### Adam fails 



- Muon: claim to be faster than AdamW. 
- The algorithm involves Newton-Schulz iteration.

The first step is: 
- first write down the algorithm in terms of matrix (linear algebra). 
- Make a quick comparison with the state-of-the-art Adam/AdamW; intuitively what is the new ingredient in the algorithm?
- How is the algorithm inspired? Any intuition? 

========
    n does Muon work well? 
This may need some simple experiments, including classification problem (logistic regression/linear regression), MLP (with multiple nonlinear layers of nonlinearity), AlexNet, ResNet, ... 
Reproduce some results in the demonstration. How does it compare to Adam/AdamW? 

Research question: 
- As we know, the AdamW/Adam does not converge on simple examples, even for convex functions. How about Muon? Does it fail even on some simple toy examples and why? 
- Is there a class of functions that Muon does work well? And the key question is: what property of the functions allow it to work? This should take two to play a tangle (two means: algorithm and the model/function).