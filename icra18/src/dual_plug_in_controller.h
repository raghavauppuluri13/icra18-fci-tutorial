// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <icra18/plug_in_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_example_controllers/pseudo_inversion.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

namespace icra18_controllers {

  struct FrankaDataContainer {
  std::unique_ptr<franka_hw::FrankaStateHandle>
      state_handle_;  ///< To read to complete robot state.
  std::unique_ptr<franka_hw::FrankaModelHandle>
      model_handle_;  ///< To have access to e.g. jacobians.
  std::vector<hardware_interface::JointHandle> joint_handles_;  ///< To command joint torques.
  double filter_params_{0.005};       ///< [-] PT1-Filter constant to smooth target values set
                                      ///< by dynamic reconfigure servers (stiffness/damping)
                                      ///< or interactive markers for the target poses.
  double nullspace_stiffness_{20.0};  ///< [Nm/rad] To track the initial joint configuration in
                                      ///< the nullspace of the Cartesian motion.
  double nullspace_stiffness_target_{20.0};  ///< [Nm/rad] Unfiltered raw value.
  const double delta_tau_max_{1.0};          ///< [Nm/ms] Maximum difference in joint-torque per
                                             ///< timestep. Used to saturated torque rates to ensure
                                             ///< feasible commands.
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;         ///< To track the target pose.
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;  ///< Unfiltered raw value.
  Eigen::Matrix<double, 6, 6> cartesian_damping_;           ///< To damp cartesian motions.
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;    ///< Unfiltered raw value.
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;               ///< Target joint pose for nullspace
                                                            ///< motion. For now we track the
                                                            ///< initial joint pose.
  Eigen::Vector3d position_d_;               ///< Target position of the end effector.
  Eigen::Quaterniond orientation_d_;         ///< Target orientation of the end effector.
  Eigen::Vector3d position_d_target_;        ///< Unfiltered raw value.
  Eigen::Quaterniond orientation_d_target_;  ///< Unfiltered raw value.
};

class DualPlugInController : public controller_interface::MultiInterfaceController<
                                   franka_hw::FrankaModelInterface,
                                   hardware_interface::EffortJointInterface,
                                   franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

   std::map<std::string, FrankaDataContainer>
      arms_data_;

  // Force control PI
  double desired_force_{0.0};
  double target_force_{0.0};
  double k_p_{0.0};
  double k_i_{0.0};
  Eigen::Matrix<double, 6, 1> force_ext_initial_;
  Eigen::Matrix<double, 6, 1> force_error_;
  static constexpr double kDeltaTauMax{1.0};

  // Wiggle motions
  double time_{0.0};
  double wiggle_frequency_x_{0.5};
  double wiggle_frequency_y_{0.5};
  double amplitude_wiggle_x_{0.1};
  double amplitude_wiggle_y_{0.1};

  // Cartesian impedance
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;

  // Dynamic reconfigure
  double target_k_p_{0.0};
  double target_k_i_{0.0};
  double wiggle_frequency_x_target_{0.5};
  double wiggle_frequency_y_target_{0.5};
  double amplitude_wiggle_x_target_{0.1};
  double amplitude_wiggle_y_target_{0.1};
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  double filter_params_{0.005};
  void updateDynamicReconfigure();

  std::unique_ptr<dynamic_reconfigure::Server<icra18::plug_in_paramConfig>>
      dynamic_server_plug_in_param_;
  ros::NodeHandle dynamic_reconfigure_plug_in_param_node_;
  void plugInParamCallback(icra18::plug_in_paramConfig& config,
                               uint32_t level);

                                 Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const FrankaDataContainer& arm_data,
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);  // NOLINT (readability-identifier-naming)

  /**
   * Initializes a single Panda robot arm.
   *
   * @param[in] robot_hw A pointer the RobotHW class for getting interfaces and resource handles.
   * @param[in] arm_id The name of the panda arm.
   * @param[in] joint_names The names of all joints of the panda.
   * @return True if successful, false otherwise.
   */
  bool initArm(hardware_interface::RobotHW* robot_hw,
               const std::string& arm_id,
               const std::vector<std::string>& joint_names);

  /**
   * Computes the decoupled controller update for a single arm.
   *
   * @param[in] arm_data The data container of the arm to control.
   */
  void updateArm(FrankaDataContainer& arm_data);

  /**
   * Prepares all internal states to be ready to run the real-time control for one arm.
   *
   * @param[in] arm_data The data container of the arm to prepare for the control loop.
   */
  void startingArm(FrankaDataContainer& arm_data);
};

}  // namespace icra18_controllers
