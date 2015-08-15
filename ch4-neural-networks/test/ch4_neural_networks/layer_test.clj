(ns ch4-neural-networks.layer-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.layer :refer :all]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(deftest feed-test
  (testing "should feed"
    (let [w [[1 1] [2 2]]
          x [[1 1] [1 1]]]
      (is (= (feed w x sigmoid) (map sigmoid '(2 4)))))))

(deftest backprop-test
  (testing "should backprop"
    (let [w [[1 1] [2 2]]
          v [2 2]
          s [1 1]]
	  (is (= (backprop w v s dsigmoid) '(-4.0 -8.0))))))

(deftest make-layer-test
  (testing "should make layer"
    (let [l (make-layer 2 2)
          w (:weights l)]
      (is (= (instance? clojure.lang.PersistentVector w)))
      (is (= (instance? clojure.lang.PersistentVector (first w))))
      (is (= (count w) 2))
      (is (= (count (first w)) 2))
      (is (= (:activations l) [0.0 0.0]))
      (is (= (:errors l) [0.0 0.0])))))
