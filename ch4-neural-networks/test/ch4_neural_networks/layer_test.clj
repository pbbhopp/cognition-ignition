(ns ch4-neural-networks.layer-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.layer :refer :all]))

(deftest feed-test
  (testing "should feed"
    (let [w [[1 1] [2 2]]
          x [[1 1] [1 1]]]
      (is (= (feed w x) '(2 4))))))
