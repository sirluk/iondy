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