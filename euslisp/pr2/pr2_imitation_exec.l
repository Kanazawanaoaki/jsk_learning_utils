#!/usr/bin/env roseus

(load "package://imitation_learning_pr2/euslisp/pr2_utils.l")

(unless (boundp '*ri*)
  (progn
    (require :pr2-interface "package://pr2eus/pr2-interface.l")
    (pr2-init)
    (setq *co* (make-coords))
    (objects (list *pr2*))
    )
  )

;; for ros
;; angle-vector topic load-ros-manifest

;; subscriber


(defun initial-pose (&key (r-sholder-pan-ja -122))
  (send *pr2* :reset-manip-pose)
  (send *pr2* :r_shoulder_pan_joint :joint-angle r-sholder-pan-ja)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  (send *viewer* :draw-objects)

  (send *ri* :stop-grasp)
  )

(defun initial-pose-tmp (&key (offset 90))
  (send *pr2* :reset-manip-pose)
  (send *co* :newcoords (make-coords :pos #f(550 150 1000) :rpy #f(0 0 -1.571)))
  (send *pr2* :larm :inverse-kinematics
        *co*
        :rotation-axis t
        :revert-if-fail nil)
  (setq lav (send *pr2* :larm :angle-vector))
  (setq dif-for-invert-vec (scale 2 (float-vector (elt lav 0) 0 (elt lav 2) 0 (elt lav 4) 0 (elt lav 6))))
  (setq n-lav (v- lav dif-for-invert-vec))
  (setq dif-for-adj-vec (float-vector offset 0 0 0 0 0 0))
  (setq rav (v- n-lav dif-for-adj-vec))

  (send *pr2* :rarm :angle-vector rav)
  (send *pr2* :head-neck-y :joint-angle 0.0)
  (send *pr2* :head-neck-p :joint-angle 60.0)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  (send *viewer* :draw-objects)

  (send *ri* :stop-grasp)
  )

(defun initial-pose-rcup ()
  (setq *initial-av* #f(299.873 75.0006 49.9994 110.003 -110.003 -20.0005 -10.0016 -10.0014 -24.3784 46.5655 -44.1727 -118.165 168.29 -67.9053 -241.519 0.0 60.0))
  (send *pr2* :angle-vector *initial-av*)
  (send *ri* :angle-vector (send *pr2* :angle-vector))
  )

;; initial-pose-tmp -> move with imitation
(defun imitation-cmd-cb (msg)
  (setq *msg* msg)
  (setq *now-cmd* (send *msg* :data))
  (format t "now angle-vector ~A~%" *now-cmd*)
  )


(defun start-imitation ()
  ;; (send *ri* :speak-jp "学習した動作を実行します")
  (ros::subscribe "/imitation/command" std_msgs::Float64MultiArray #'imitation-cmd-cb)
  (setq *now-cmd* (send *pr2* :rarm :angle-vector))
  (ros::rate 10);;10
  (do-until-key
   ;; subscribeして動作する．
   (send *pr2* :head-neck-y :joint-angle 0.0)
   (send *pr2* :head-neck-p :joint-angle 60.0)
   (send *pr2* :rarm :angle-vector *now-cmd*)
   ;; (send *ri* :angle-vector (send *pr2* :angle-vector))
   (send *ri* :angle-vector-sequence (list (send *pr2* :angle-vector)) (list 400))

   ;; (send *ri* :wait-interpolation nil 0.01)
   (send *viewer* :draw-objects)
   (ros::spin-once)
   (ros::sleep)
   )
  )
