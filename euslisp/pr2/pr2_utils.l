#!/usr/bin/env roseus

;; (unless (boundp '*ri*)
;;   (progn
;;     (require :pr2-interface "package://pr2eus/pr2-interface.l")
;;     (pr2-init)
;;     )
;;   )

(ros::roseus "pr2_utils")

;; dialog
(ros::load-ros-manifest "speech_recognition_msgs")
(setq *data-flag* nil)

(defun voice-cb (msg)
  (when msg
    (setq *msg* msg)
    (setq *voice-data* (send *msg* :transcript))
    (setq *data-flag* t)
    (format t "data: ~A ~%" *data*)
    )
  )

(defun dialogue-func ()
  (ros::unsubscribe "/speech_to_text")
  (setq *data-flag* nil)
  (ros::subscribe "/speech_to_text" speech_recognition_msgs::SpeechRecognitionCandidates #'voice-cb)
  (until *data-flag*
         (ros::spin-once)
         (ros::sleep)
         )
  (send *ri* :speak-jp "ありがとうございます")
  (unix:sleep 1)
  )

(defun ask-end ()
  (let (end-flag)
    (send *ri* :speak-jp "データ収集を終了しますか？")
    (dialogue-func)
    (setq ask-ans *voice-data*)
    (if (equal ask-ans '("はい"))
        (setq end-flag t)
      (setq end-flag nil)
      )
    (send *ri* :speak-jp "了解しました")
    end-flag
    ))

(defun voice-ok-cb (msg)
  (when msg
    (setq *msg* msg)
    (setq *data* (send *msg* :transcript))
    (if (equal *data* '("OK"))
        (progn
          (format t "了解しました！ ~%" *data*)
          (setq *ok-flag* t)
          )
      (format t "data: ~A ~%" *data*)
      )
    )
  )

(defun ok-wait (&key (w-time 2))
  (ros::unsubscribe "/speech_to_text")
  (setq *ok-flag* nil)
  (ros::subscribe "/speech_to_text" speech_recognition_msgs::SpeechRecognitionCandidates #'voice-ok-cb)
  (until *ok-flag*
         (ros::spin-once)
         (ros::sleep)
         )
  (send *ri* :speak-jp "ありがとうございます")
  (unix:sleep w-time)
  )


;; manequine
(ros::load-ros-manifest "pr2_mechanism_msgs")
;; (ros::wait-for-service "/pr2_controller_manager/switch_controller")

(defun manequine-mode (&key (arm (list "rarm")) (start t))
  (let ((req (instance pr2_mechanism_msgs::SwitchControllerRequest :init))
	(normal-controllers (list))
	(loose-controllers (list))
	)
    (dolist (e arm)
      (if (equal e "rarm")
	  (progn
	    (setq normal-controllers (append normal-controllers (list "r_arm_controller")))
	    (setq loose-controllers (append loose-controllers (list "r_arm_controller_loose")))
	    )
	)
      (if (equal e "larm")
	  (progn
	    (setq normal-controllers (append normal-controllers (list "l_arm_controller")))
	    (setq loose-controllers (append loose-controllers (list "l_arm_controller_loose")))
	    )
	))
    (if start
	(progn
	  (send req :start_controllers loose-controllers)
	  (send req :stop_controllers normal-controllers)
	  (format t "start ~A manequine mode! ~%" arm)
	  )
      (progn
	(send req :start_controllers normal-controllers)
	(send req :stop_controllers loose-controllers)
	(format t "stop ~A manequine mode! ~%" arm)
	)
      )
    (setq res (ros::service-call "/pr2_controller_manager/switch_controller" req))
    (format t "controller service response: ~A ~%" (send res :ok))
    ))

(defun enable-gripper-manequine-mode (&key (arm (list "rarm")) (start t))
  (let ((req (instance pr2_mechanism_msgs::SwitchControllerRequest :init))
	(normal-controllers (list))
	(loose-controllers (list))
	)
    (dolist (e arm)
      (if (equal e "rarm")
	  (progn
	    (setq normal-controllers (append normal-controllers (list "r_gripper_controller")))
	    (setq loose-controllers (append loose-controllers (list "")))
	    )
	)
      (if (equal e "larm")
	  (progn
	    (setq normal-controllers (append normal-controllers (list "l_gripper_controller")))
	    (setq loose-controllers (append loose-controllers (list "")))
	    )
	))
    (send req :start_controllers loose-controllers)
    (send req :stop_controllers normal-controllers)
    (format t "start ~A manequine mode! ~%" arm)
    (setq res (ros::service-call "/pr2_controller_manager/switch_controller" req))
    (format t "controller service response: ~A ~%" (send res :ok))
	(send req :start_controllers normal-controllers)
	(send req :stop_controllers loose-controllers)
	(format t "stop ~A manequine mode! ~%" arm)
    (setq res (ros::service-call "/pr2_controller_manager/switch_controller" req))
    (format t "controller service response: ~A ~%" (send res :ok))
    ))

;; data_collection
(ros::load-ros-manifest "std_srvs")

(defun call-service (service req)
  (if (ros::wait-for-service service 0)
      (ros::service-call service req)
    nil))

(defun start-data-collection (&key (service "data_collection_rosbag_test/start_request"))
  (setq req (instance std_srvs::TriggerRequest :init))
  (setq res (ros::service-call service req t))
  )

(defun end-data-collection (&key (service "data_collection_rosbag_test/end_request"))
  (setq req (instance std_srvs::TriggerRequest :init))
  (setq res (ros::service-call service req t))
  )

;; use yaml config
(defun load-yaml-config (config-file)
  (let ((control_joints nil))
    (unix:system (format nil "rosparam load ~A yaml_load_eus" config-file))
    (setq control_joints (ros::get-param "/yaml_load_eus/control_joints"))
    (setq dim_list (ros::get-param "/yaml_load_eus/dim_list"))
    (setq initial_joints (ros::get-param "/yaml_load_eus/initial_joints"))
    (format t "~%load params from ~A ~%" config-file)
    (format t " now control_joints is ~A ~%" control_joints)
    (format t " now dim_list is ~A ~%" dim_list)
    (format t " now initial_joints is ~A ~%" initial_joints)
    (list control_joints dim_list initial_joints)))

(defun load-config-from-name (config-name)
  (let ((file-path nil))
    (setq file-path (ros::resolve-ros-path (format nil "package://jsk_learning_utils/configs/~A" config-name)))
    (format t "now path is ~A ~%" file-path)
    (load-yaml-config file-path)
    ))
