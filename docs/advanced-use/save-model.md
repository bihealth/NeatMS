# Save neural network models

We are almost done, and this part is very simple! So now that we have our newly trained model, we know that threshold that we should use, all we have to do is save it for later use.

Here is the way to that:

``` python
# Good practice is to set the threshold that should be use in the filename 
# so you don't lose it
nn_handler.class_model.save('my_own_model_020.h5')
```  

One last important note: If you have changed some parameters such as the margin or the matrix size, remember that you will have to set those same parameters when creating the `NN_handler` object every time you want to use this model. Otherwise, NeatMS will attempt to feed peaks with wrong sizes to the model.

You now know everything there is to know about NeatMS, remember, you can report bugs or make feature request on the github repository. Do not hesitate to give us feedback, that would help us very much improve the tool.

Thanks for using NeatMS!