(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

(deftest make-neural-network-test
  (testing "making an initial state neural network"
    (let [nn [(make-layer 4 2) (make-layer 1 2)]]
      (is (= (count nn) 2)))))

