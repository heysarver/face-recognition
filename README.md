# face-recognition
Python script to train a model on a set of images of a person's face then detect if that person is in other images.

```bash
$ python app.py
Epoch 1, Loss: 9.270924538373947
Epoch 2, Loss: 3.2445448637008667
Epoch 3, Loss: 2.717623680830002
...
Epoch 18, Loss: 0.6142306160181761
Epoch 19, Loss: 0.5572632923722267
Epoch 20, Loss: 0.5132462279871106
Training complete.
Image: test_images/52033290811_1552c118e5_o.jpg, Label: person, Score: 0.992526650428772
Image: test_images/2564439515_7871d82f93_o.jpg, Label: person, Score: 0.9991218447685242
Image: test_images/2656029169_10bf259b5a_o.jpg, Label: person, Score: 0.9990962743759155
Image: test_images/52079640414_ce90c759f5_o.jpg, Label: not_person, Score: 0.5172777771949768
Image: test_images/2730165138_630dc9e268_o.jpg, Label: person, Score: 0.9950997233390808
Image: test_images/2580427039_483bc48d9c_b.jpg, Label: person, Score: 0.999819815158844
Image: test_images/2455746450_4a5f4965ff_b.jpg, Label: person, Score: 0.9991812109947205
```
