(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

(deftest make-neural-network-test
  (testing "making an initial state neural network"
    (let [nn [(make-layer 4 2) (make-layer 1 2)]]
      (is (= (count nn) 2)))))

(def wi [[ 0.6888437030500962  0.515908805880605   -0.15885683833831] 
         [-0.4821664994140733  0.02254944273721704 -0.19013172509917142] 
         [ 0.5675971780695452 -0.3933745478421451  -0.04680609169528838]])

(def wo [[0.1667640789100624] [0.8162257703906703] [0.009373711634780513]])

(def training-input
  [[[0 0] [0]]
   [[0 1] [1]]
   [[1 0] [1]]
   [[1 1] [0]]])
