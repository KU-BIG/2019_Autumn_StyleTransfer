## Description


#### style_transfer_pytorch.ipynb
- 아래의 링크의 pytorch tutorials를 따라하면서 style transfer for images를 연습한 파일입니다. 

- https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#underlying-principle

#### style_transfer_hyperparameter.ipynb
- 위의 tutorial을 따라하면서 style loss의 hyper parameter를 변경해서 테스트 해 본 파일입니다. 
- style loss hyperparameter 인 \beta를  10^6에서 10^2로 변경해서 진행해 본 결과 작을수록 원본 이미지에 비슷해 지는 것을 확인했습니다. 

#### style_transfer_contentloss.ipynb
- 위의 tutorial 중 content loss를 구하기 위해서 higher convolution layer의 activation을 사용하였는데, 그것을 lower convolution layer로 바꾸어 테스트 해 보았습니다. 
- 기존의, 뒤의 convolution layer를 사용할 때 보다 초반부의 convolution layer를 사용했을 경우 content loss는 값은 더 작게 나타나고, 결과물이 조금 더 자세해 짐을 확인할 수 있었습니다. 

#### gatys_tutorial_styleLoss_modification1.ipynb
- 논문에서는 style loss를 구할 때 각각 layer마다 weight를 고려하였는데, tutorial에서는 이 weight를 고려하지 않고 output image를 생성하여 논문의 방식대로 styleLoss를 구해보았습니다.
- 논문에 다르면 l번째 레이어의 weight인 w_l은 1 나누기 style layer의 개수와 같습니다. 즉 모든 레이어의 weight의 값이 다 같습니다.
- 따라서 layer에 weight를 추가하는 것은 totalLoss를 구할 때 hyperparameter인 beta를 수정하는 것과 같습니다. 
