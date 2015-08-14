(ns ch4-neural-networks.layer-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.layer :refer :all]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(deftest feed-test
  (testing "should feed"
    (let [w [[1 1] [2 2]]
          x [[1 1] [1 1]]]
      (is (= (feed w x sigmoid) (map sigmoid '(2 4)))))))
