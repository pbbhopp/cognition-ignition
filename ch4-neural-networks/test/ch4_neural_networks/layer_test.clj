(ns ch4-neural-networks.layer-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.layer :refer :all]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(def l (->Layer [[1 1] [2 2]] [0.0 0.0] [0.25 0.5] [0.1 0.1]))

(deftest feed-test
  (testing "should feed"
    (let [x [1 1]
          l (feed l x sigmoid)]
      (is (= (:activations l) (map sigmoid '(2 4)))))))

(deftest backprop-test
  (testing "should backprop"
    (let [v [2 2]
          s [1 1]
          l (backprop l v s dsigmoid)]
	  (is (= (:deltas l) '(-4.0 -8.0))))))

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

(deftest update-weights-test
  (testing "should update weights"
    (let [out [5 5]
          rate 0.1
          layer (update-weights l out rate)
          w (:weights layer)]
      (is (= w [[1.125 1.125] [2.25 2.25]])))))
      
