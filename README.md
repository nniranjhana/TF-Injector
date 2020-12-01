<h1 align="center"> 
  <img src="https://user-images.githubusercontent.com/29974283/100801172-78f3d700-33dc-11eb-8756-375ddbd740ca.png" height="150">
  <br/>
    TF-Injector
  </br>
</h1>

<h2 align="center">A Versatile Fault Injector for ML Applications</h2>

TF-Injector is a software fault injector for TensorFlow v2 applications written in Python 3.
A variety of faults are supported at the data, hardware and software levels.

### 1. Dependencies

1. TensorFlow framework (v2.0 or greater)

2. Python (v3 or greater)

3. PyYaml (v3 or greater)

4. numpy package (part of TensorFlow)

### 2. Installation and runs

Following are the installation and usage instructions for a Linux platform.

1. Clone the repository.

    ```
    git clone https://github.com/nniranjhana/TF-Injector.git
    ```

2. Set the python path for the TF-Injector project so that it can be executed from other scripts. You can also add it permanently to .bashrc if you prefer.

    ```
    export PYTHONPATH=$PYTHONPATH:$TF-INJECTORHOMEPATH
    ```

	where `$TF-INJECTORHOMEPATH` might be like `/home/nj/TF-Injector`

3. You can navigate to [conf/](https://github.com/nniranjhana/TF-Injector/tree/master/conf) to check out how to set the fault injection configuration for the tests you plan to run.

4. Let's see an example of how to mutate the weights of a model. Go to [tests/tfv2/model](https://github.com/nniranjhana/TF-Injector/tree/master/tests/tfv2/model) and set the sample.yaml file in [tests/tfv2/model/confFiles](https://github.com/nniranjhana/TF-Injector/tree/master/tests/tfv2/model/confFiles). If you are running from the examples in this directory, this is the file that gets picked up.

5. Run the test to observe the fault injection. For example, let's say we run the simple neural network example:

    ```
    python nn-mnist.py
    ```

    with the following configuration:

    ```
    Artifact: 0
    Type: mutate
    Amount: 1
    Bit: 30
    ```

    This means that a single bit is flipped (30th bit position) in the model parameter tensor that holds the first hidden layer weights.

    The results for this run are:

    ```
    nj@nj-arch-xps:~/TF-Injector/tests/tfv2/model$ python nn-mnist.py

    WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.

    2020-05-06 17:02:42.180771: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

    step: 100, loss: 2.033674, accuracy: 0.578125
    step: 200, loss: 1.827772, accuracy: 0.738281
    step: 300, loss: 1.724117, accuracy: 0.808594
    step: 400, loss: 1.675569, accuracy: 0.843750
    step: 500, loss: 1.646916, accuracy: 0.859375
    Test accuracy before injections: 0.829200

    Test accuracy after injections: 0.734800
    ```
