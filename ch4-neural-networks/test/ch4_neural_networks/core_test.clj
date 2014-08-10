(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

(deftest make-neural-network-test
  (testing "making an initial state neural network"
    (let [nn (make-neural-network 2 2 1)]
      (is (= (:hidden-nodes @nn) [[1 1 1]]))
      (is (= (:input-nodes @nn) [[1 1 1]]))
      (is (= (:output-nodes @nn) [[1]])))))

(def wi [[ 0.6888437030500962  0.515908805880605   -0.15885683833831] 
         [-0.4821664994140733  0.02254944273721704 -0.19013172509917142] 
         [ 0.5675971780695452 -0.3933745478421451  -0.04680609169528838]])

(def wo [[0.1667640789100624] [0.8162257703906703] [0.009373711634780513]])

(deftest activate-nodes-test
  (is (= (activate-nodes [0 0 1] wi sigmoid) [0.5135924495204157 -0.37426574501728027 -0.046771940534449614])))

(def neural-network (make-neural-network 2 2 1))

(swap! neural-network assoc :input-weights wi)

(swap! neural-network assoc :output-weights wo)

(deftest feed-forward-test
  (testing "feed forward activating nodes in neural network"
    (feed-forward neural-network [0 0])
    (is (= (:hidden-nodes @neural-network) [[0.5135924495204157 -0.37426574501728027 1]]))
    (is (= (:output-nodes @neural-network) [[-0.20740949312127344]]))))

(deftest back-propagate-test
  (testing "back propogation in neural network to find errors and then update weights"
    (back-propagate neural-network [[0 0] [0]] 0.5)
    (is (= (:input-weights @neural-network) 
      [[0.217734792922362] [0.7790823266510097] [0.10861721503888083]]))
    (is (= (:output-weights @neural-network) 
      [[ 0.6888437030500962  0.515908805880605   -0.15885683833831] 
       [-0.4821664994140733  0.02254944273721704 -0.19013172509917142] 
       [ 0.5797818504506633 -0.3237162205844442  -0.04680609169528838]]))))

(def training-input
  [[[0 0] [0]]
   [[0 1] [1]]
   [[1 0] [1]]
   [[1 1] [0]]])

(deftest training-test
  (testing "training of neural network with full training data set"
    (let [nnet (make-neural-network 2 2 1)]
      (swap! nnet assoc :input-weights wi)
      (swap! nnet assoc :output-weights wo)
      (train nnet training-input 0.5)
      (is (= (:input-weights @nnet) 
        [[0.21650107466220375] [0.5578574104610892] [0.5785890097427964]]))
      (is (= (:output-weights @nnet) 
        [[ 0.6787406035661884  0.5012396912570095  -0.15885683833831]
         [-0.37647384745464746 0.3843265480669719  -0.19013172509917142] 
         [ 0.68691868299303    0.04891969157308979 -0.04680609169528838]])))))
