# Media credits

Body footage:

- [Talented urban woman dancing at night on a basketball court](https://mixkit.co/free-stock-video/talented-urban-woman-dancing-at-night-on-a-basketball-court-51314/), Mixkit, under the [Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree).
- [A Young Girl Dancing In The Street](https://www.pexels.com/video/street-girl-dancing-dance-4540332/), olia danilevich, under the [Pexels License](https://www.pexels.com/license/).
- [Boy dancing a lively choreography in the street](https://mixkit.co/free-stock-video/boy-dancing-a-lively-choreography-in-the-street-51320/), Mixkit, under the [Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree).

Body media uses Body Pose `body_s240000.ckpt`, native77 final775 Motion,
ViTPose random-blur epoch 10 observations and SOMA 0.3.3, with Contact/IK
postprocessing for temporal Motion. The released checkpoint bundle uses the
original ViTPose instead. G1 motion is retargeted with
[SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter).

Hand media comes from the EgoHand 1.0-GA showcase; it is not a new inference run
of the selected `egohand_s060000.ckpt` checkpoint.
