(ns ch6-naive-bayes.core)

(defn make-classifier []
  (atom {:data {} :counter {}}))

(defn increment-feature [classifier feature category]
  (let [result  ((keyword feature) (:data @classifier) {category 0})
        updated (merge-with + {category 1} result)]
    (swap! classifier assoc-in [:data (keyword feature)] updated)))

(defn increment-category [classifier category]
  (let [result  (category (:counter @classifier) 0)]
    (swap! classifier assoc-in [:counter category] (inc result))))

(defn feature-probability [classifier feature category]
  (let [count (category ((keyword feature) (:data @classifier)))
        div   (category (:counter @classifier))]
    (if (= div 0)
      0
      (/ count div))))

(defn category-probability [classifier category]
  (let [count (category (:counter @classifier))
        div   (reduce + (vals (:counter @classifier)))]
    (if (= div 0)
      0
      (/ count div))))

(defprotocol Feature
  (get-words [str]))

(extend-type String
  Feature
  (get-words [str] (vec (clojure.string/split str #" "))))

(defn train [classifier terms category]
  (increment-category classifier category)
  (doseq [word terms]
    (increment-feature classifier word category)))
