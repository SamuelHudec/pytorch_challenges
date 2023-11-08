# Pytorch Challenges

This project was created as an initiative to kick myself for deeper understanding of the pytorch library. 
I asked my skilled colleague to create challenges for me. Knowledge of programming and mathematics is required.

Each task is specified in a `template.py` file in challenge folder. Based on it, I should be able to compose a solution.
Remember template is only __hint file__. Of course every one can use his creativity to deliver running solution.

## First Challenge (MLP)

This one is simple as possible. Code `MultiLayerPerceptron` that means simple neural network.
I found as challenge is to create neural network by list, use different activation function and apply batch normalisation. 
Also challenge learns me to create fake data, what is baseline for test composed `nn.Module` model. 
When I finished task I can play with some famous data as `CIFAR10` and more.

## Second Challenge (Attention)

Now lets move further straight to NLP basics. What is more basic as Self Attention Head? If you understand attention, 
than you will understand core of transformers as well. Challenge trains me code just model from paper equation. 
Narutaly, every one can continue with multi head attention as I did (universal also for decoder) and finish with 
transformer (on my list). Multi haead attention have more advanced solution using also batch size and also ready 
for decoder. This solution can be used in transformer.

## Third Challenge (Product2vec)

Let's continue with creating model moduls from paper as before. Situation is now different, no examples as with MLP 
and NLP tasks, that means I have to rely on myself. This challenge will touch recommendations with famous P-companion alg.
For simplicity `Product2vec` from section 4.1 is enough. Key on task is to understand pipeline inputs and outputs they 
propose. Mostly is described in figure 4. Attention head in graphs is bigger challenge not because of code, but 
understanding role of message passing in attention mask. I like their training objective i.e. Loss function based 
on hinge loss. if you want to make it more interesting, add loss function.

## My best practices
* always develop in classical script not in jupyter notebook. This leads to the smoothness of the solution.
* jupyter notebook is powerful, but use that power properly. I develop in scripts and using jupyter just for testing model parts.

__Anybody who have an idea for this challenge please do not hesitate to share. Challenge is accepted in advance.__