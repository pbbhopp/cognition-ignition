(ns ch4-neural-networks.layer)
  
(defn feed [w x f]
  (let [coll (map #(reduce + (map (fn [a b] (* a b)) %1 %2)) w x)]
    (map f coll)))
