---
title: "Adverserial Training"
date: "2022-08-22"
description: "Implementing gradient reversal and adversarial heads to study and reduce bias in language models."
category: "Research"
tags: ["bert", "artificial-intelligence", "nlp", "fairness", "machine-learning"]
original_url: "https://medium.com/@lukas.hauzenberger/adverserial-training-5bb5ea919ae7"
---

Large language models following the transformer architecture have been incredibly successful since their introduction by Vaswani et. al. in 2018. They have given rise to large pre-trained models such as BERT (Devlin et. al., 2019) which can be finetuned and applied to a wide variety of language tasks. BERT provides rich text encoding vectors due to being trained on millions of rows of unlabeled text. There is many great articles on Transformers and BERT on the web so I wont go into further detail here.



## Bias in Machine Learning



![](<images/3f9ab07c08b4548b.png>)



The way BERT is trained however is a gift and a curse. On the one hand training is unsupervised, therefore theoretically it can be fed an infinite number of text elements without any manual data labelling. In practice BERT was still trained on extremely large copora (English Wikipedia with 2,500M words and BooksCorpus with 800M words) which lead to the model becoming incredibly good at language understanding. On the other hand however “unsupervised” also means different types of bias that exist in a language such as race or gender bias can find their way into the model without any data quality checks. Various researchers such as Buolamwini & Gebru (Gender Shades study) have already shown that bias does in fact exist in modern Machine Learning systems and a number of methods have been introduced to combat bias at all stages of the Machine Learning workflow. Today I would like to shine the spotlight on one of them: Adverserial Training



## Adverserial Training



Adverserial Training emerged from the field of domain adaptation where the technique was first described by Ganin & Lempitsky (2015). The basic idea is to reduce the impact of a pre-defined attribute (such as gender) on task predictions. To do that intermediate model outputs (such as BERT encodings) are fed into a task head as well as a second head which has the objective to predict this attribute. During the backward pass the sign of the gradients flowing back out of this head is reversed (in the gradient reversal layer or **GRL** for short). This has the effect of “punishing” the rest of the model for enabling correct domain predictions.



![](<images/79c6a91d9461c0f0.png>)

*adverserial training architecture*



Adverserial training provides an elegant way to finetune a pre-trained language encoder model such as BERT while controlling for bias with respect to a specific attribute.



## Pytorch Implementation



In the following section I will show you step by step how to implement the building blocks required for adversarial training in pytorch



- Dataset
- Gradient Reversal Layer
- Adverserial head
- Adjusted update step
- Evaluation by adversarial attack



All the code examples are taken from this [**repository** ](<https://github.com/sirluk/adverserial_training>)which I created to showcase adversarial training.



### Dataset



I used the BIOS dataset (De-Arteaga et. al. 2019). It includes short biographies with two labels job title and gender. One datapoint contains the following information:



- BIO: “dr practices at sai ram super speciality dental clinic in malkajgiri hyderabad completed bds from dr ntr university of health sciences andhra pradesh in”
- TITLE: “dentist”
- GENDER: “F”



Here title is the main classification objective and gender acts as the protected attribute. More information on how to get the data can be found in the [README ](<https://github.com/sirluk/adverserial_training/blob/main/README.md>)of the git repository.



### Gradient Reversal Layer



Before we can build the adversarial head we need a simple class for the Gradient Reversal Layer which inherits from the autograd function class. Autograd functions enable you to define new operations for the forward and backward pass. In our case we only want to reverse the sign of the gradients flowing back out of the adversarial head in the backward pass which we achieve in the backward method as follows



```python
from torch.autograd import Function

class ReverseLayerF(Function):
    # https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/2
    @staticmethod
    def forward(ctx, input_, lmbda):
        ctx.lmbda = lmbda
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lmbda
        return grad_input, None
```



Lambda here is an optional hyperparameter which we could use to also adjust the magnitude of the gradients. In my experiments however at best this did not have any effect, more commonly however it even reduced the effectiveness of adversarial training.



**NOTE:** A special quirk of autograd functions is that if you define no operation in the forward method the function does not modify the gradients in the backward method. For this reason we apply .view\_as() which effectively leaves the input unchanged but still gets registered as an operation.



### Adversarial Head



We can now use the GRL class to build the adverserial head. To keep focus on the important code snippets our adverserial head class inherits from a generic classifier head. I assume you are familiar with such heads but in the file [model\_heads.py ](<https://github.com/sirluk/adverserial_training/blob/main/src/models/model_heads.py>)in the repository you can find one simple and one more advanced version. Furthermore a common practice for the adverserial head is to have several classifiers side by side with loss averaging to increase robustness. You can see this in the \_init\_ method of the AdvHead class where I create a list of classifiers.



```python
  def __init__(self, adv_count: int = 1, **kwargs):
      super().__init__()    
      self.heads = nn.ModuleList()
      for i in range(adv_count):
          self.heads.append(ClfHead(**kwargs))
```



In the forward pass we iterate over the list of classifiers and pass the input (i.e. the BERT embeddings) to each of them. Before we do this however we apply the GRL class to the input. I put this in a separate method (forward\_reverse) which wraps around the forward method because in some cases we might not need this functionality (for example at inference time). Here you can see the full AdvHead class:



```python
class AdvHead(nn.Module):
    
    def __init__(self, adv_count: int = 1, **kwargs):
        super().__init__()    
        self.heads = nn.ModuleList()
        for i in range(adv_count):
            self.heads.append(ClfHead(**kwargs))
            
    def forward(self, x):
        out = []
        for head in self.heads:
            out.append(head(x))
        return out
            
    def forward_reverse(self, x, lmbda = 1.):
        x_ = ReverseLayerF.apply(x, lmbda)
        return self(x_)
    
    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()
```



### Adjusted update step



In the file [adv\_model.py](<https://github.com/sirluk/adverserial_training/blob/main/src/models/model_adv.py>) you can see how adversarial training is integrated into a classification model. One thing we need to do is to combine the loss from the task head with the loss from the adversarial head.



```python
loss = 0.

inputs, labels_task, labels_protected = batch
inputs = dict_to_device(inputs, self.device)
hidden = self._forward(**inputs)
outputs_task = self.task_head(hidden)
loss_task = loss_fn(outputs_task, labels_task.to(self.device))
loss += loss_task

outputs_protected = self.adv_head.forward_reverse(hidden, lmbda = adv_lambda)
loss_protected = self._get_mean_loss(outputs_protected, labels_protected.to(self.device), loss_fn_protected)
loss += loss_protected

loss.backward()
```



Furthermore I mentioned earlier that typically the adversarial head consists of several classifiers. Therefore we need a function to average the outputs before running the backward pass. This is what self.\_get\_mean\_loss() does



```python
  def _get_mean_loss(self, outputs: Union[torch.Tensor, List[torch.Tensor]], labels: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
      if isinstance(outputs, torch.Tensor):
          outputs = [outputs]
      losses = []
      for output in outputs:
          losses.append(loss_fn(output, labels))
      return torch.stack(losses).mean()    
```



For evaluation however we do not average the loss but pick the mode of the predictions (majority vote).



**NOTE:** In general when using accuracy as the evaluation metric we strive to reach 50% accuracy on the protected attribute meaning random predictions i.e. no information about the attribute is contained in the model.



### Evaluation by adversarial attack



We now already went over the essential components to make adversarial training work. The only thing left is to talk about evaluating how effective it is in actually removing information about the protected attribute from the encodings which are being fed into the classification head. A common thing to do is to run an adversarial attack. For this you first freeze the encoder model and with it the encodings. With a newly initialized adverserial head you then try to predict the protected attribute (This is now also a use case where we do not need the forward\_reverse method from the adversarial head and can use the forward method). The idea is to show how much information about the protected attribute is still contained in the embeddings after adversarial training.



For the BIOS dataset I ran two experiments which can be found [**here**](<https://tensorboard.dev/experiment/RoupPbp9SoCEeC6BOEdwXA/#scalars>)



- experiment **with adverserial training**:  
  adverserial-bert\_uncased\_L-2\_H-128\_A-2–64–2e-05-seed0
- experiment **without adverserial trainin**g:  
  task-bert\_uncased\_L-2\_H-128\_A-2–64–2e-05-seed0



When looking at the results you can see that the metric “acc\_adv\_attack” is considerably lower for the model trained with adversarial training (orange line). At the same time task accuracy stays at a similar level.



![](<images/722a5206ca075aa8.png>)



## Final thoughts / things to consider



While adversarial training can work quite well there is some things to consider before applying it



- One thing to consider is the time investment needed for data labelling. Adversarial training requires additional labels for each protected attribute we want to remove / control.
- A second thing to take care of is to ensure that the training dataset is relatively balanced in terms of task and protected labels. This means that there should exist equal amount of each protected class within each task class. Otherwise the model could learn that one protected class is more likely overall than the others and by only predicting this one could reach an accuracy above 50%.



I hope you enjoyed this article and that it maybe helps you implement adversarial training yourself!



I already mentioned the repository with all the code in the article but since I know some of you might scroll to the bottom right away here it is again: [**link**](<https://github.com/sirluk/adverserial_training>)



## References



\[1\] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. & Polosukhin, I. (2017). Attention is All you Need. Advances in Neural Information Processing Systems. Volume 30.



\[2\] Devlin, J., Chang, M., Lee, K. & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1.



\[3\] Buolamwini, J. & Gebru, T. (2018). Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification. Proceedings of the 1st Conference on Fairness, Accountability and Transparency. Volume 81.



\[4\] Ganin, Y. & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. International conference on machine learning.
