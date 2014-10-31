(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

(def weights [[ 0.014375367932126654  0.216311916893445420  0.74139912430793380] 
               [-0.245757173379700660 -0.267149242897378800  0.15417318630866406]
               [-0.145751006387052960 -0.024118668071382188 -0.13222709112239060]
               [ 0.224316313871827040 -0.183629891976191730 -0.40702572197133036]])

(deftest recursive-forward-propagate-test
  (testing "recursively forward propagate neural network"
    (let [results     (forward weights '(0))
          activations (:activations results)
          outputs     (:outputs results)]
      (is (= activations [0.7413991243079338 0.15417318630866406 -0.1322270911223906 -0.276425077965549]))
      (is (= outputs [0.6299897352646393 0.15296315642714714 -0.13146182301041381 -0.2695931916334675])))))

