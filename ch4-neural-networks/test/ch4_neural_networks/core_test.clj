(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

(def training-input
  [[[0 0] [0]]
   [[0 1] [1]]
   [[1 0] [1]]
   [[1 1] [0]]])

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
  (is (= (activate-nodes [0 0 1] wi) [0.5135924495204157 -0.37426574501728027 -0.046771940534449614])))
