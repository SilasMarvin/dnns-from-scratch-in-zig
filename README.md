This repo is a very simple implementation of deep neural networks written in the Zig programming language.

It does not utilize tensors or any automatic differentiation library as it is only composed of fully connected feed forward layers.

To solve MNIST, create a new directory "data" in the root of the project, download the MNIST files from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/), and unzip and store the files in the "data" directory. Then run:

```
zig build run
```

If the above url is not working for you, checkout: [https://github.com/cvdfoundation/mnist](https://github.com/cvdfoundation/mnist).
