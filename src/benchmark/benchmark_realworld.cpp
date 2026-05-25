#include "tools.hpp"
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseArray.h>
#include <random>
#include <ctime>
#include <tf/transform_broadcaster.h>
#include "bavoxel.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <iomanip>
#include <malloc.h>

using namespace std;

template <typename T>
void pub_pl_func(T &pl, ros::Publisher &pub)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = ros::Time::now();
  pub.publish(output);
}

ros::Publisher pub_path, pub_test, pub_show, pub_cute;

int max_frames = 100;

string join_path(const string &base, const string &name)
{
  if(base.empty() || base.back() == '/') return base + name;
  return base + "/" + name;
}

int read_tum_pose(vector<double> &tims, vector<string> &pcd_names, PLM(3) &rots, PLV(3) &poss, string prename) {
  string posename = join_path(prename, "scan_states_odom.txt");
  ifstream inFile(posename);

  if(!inFile.is_open())
  {
    printf("open %s fail\n", posename.c_str()); return 0;
  }
  
  string lineStr, str;
  while(getline(inFile, lineStr)) {
    if(lineStr.empty() || lineStr[0] == '#') continue;
    stringstream ss(lineStr);
    vector<string> tokens;
    vector<double> nums;
    while(ss >> str) {
      tokens.push_back(str);
      nums.push_back(stod(str));
    }

    if(nums.size() < 8) {
      printf("Skip invalid pose line with %zu values: %s\n", nums.size(), lineStr.c_str());
      continue;
    }

    tims.push_back(nums[0]);
    pcd_names.push_back(tokens[0] + ".pcd");
    poss.push_back(Eigen::Vector3d(nums[1], nums[2], nums[3]));
    Eigen::Quaterniond q(nums[7], nums[4], nums[5], nums[6]);
    rots.push_back(q.normalized().toRotationMatrix());
  }
  inFile.close();

  if(tims.size() != rots.size()) {
    printf("TUM input must contain one timestamp per pose to map poses to pcd filenames.\n");
    return 0;
  }

  if (max_frames > 0 && tims.size() > static_cast<size_t>(max_frames)) {
    size_t frame_num = static_cast<size_t>(max_frames);
    size_t last_idx = tims.size() - 1;
    vector<double> sampled_tims;
    vector<string> sampled_pcd_names;
    PLM(3) sampled_rots;
    PLV(3) sampled_poss;
    sampled_tims.reserve(frame_num);
    sampled_pcd_names.reserve(frame_num);
    sampled_rots.reserve(frame_num);
    sampled_poss.reserve(frame_num);

    if(frame_num == 1) {
      sampled_tims.push_back(tims.front());
      sampled_pcd_names.push_back(pcd_names.front());
      sampled_rots.push_back(rots.front());
      sampled_poss.push_back(poss.front());
    } else {
      for(size_t i=0; i<frame_num; i++) {
        size_t idx = (i * last_idx + (frame_num - 1) / 2) / (frame_num - 1);
        sampled_tims.push_back(tims[idx]);
        sampled_pcd_names.push_back(pcd_names[idx]);
        sampled_rots.push_back(rots[idx]);
        sampled_poss.push_back(poss[idx]);
      }
    }

    cout << "Downsample TUM poses from " << tims.size() << " to " << frame_num
         << " with uniform interval sampling." << endl;
    tims.swap(sampled_tims);
    pcd_names.swap(sampled_pcd_names);
    rots.swap(sampled_rots);
    poss.swap(sampled_poss);
  }
  if(tims.empty()) return 0;
  std::cout << "first rot:\n" << rots.front() << "\nfirst time: " << std::setprecision(19) << tims.front() << endl;
  std::cout << "last rot:\n" << rots.back() << "\nlast time: " << std::setprecision(19) << tims.back() << endl;
  std::cout << "#tims " << tims.size() << ", #poses " << rots.size() << endl;
  return tims.size();
}

int read_pose(vector<double> &tims, PLM(3) &rots, PLV(3) &poss, string prename)
{
  string readname = prename + "alidarPose.csv";

  cout << readname << endl;
  ifstream inFile(readname);

  if(!inFile.is_open())
  {
    printf("open fail\n"); return 0;
  }

  int pose_size = 0;
  string lineStr, str;
  Eigen::Matrix4d aff;
  vector<double> nums;

  int ord = 0;
  while(getline(inFile, lineStr))
  {
    ord++;
    stringstream ss(lineStr);
    while(getline(ss, str, ','))
      nums.push_back(stod(str));

    if(ord == 4)
    {
      for(int j=0; j<16; j++)
        aff(j) = nums[j];

      Eigen::Matrix4d affT = aff.transpose();

      rots.push_back(affT.block<3, 3>(0, 0));
      poss.push_back(affT.block<3, 1>(0, 3));
      tims.push_back(affT(3, 3));
      nums.clear();
      ord = 0;
      pose_size++;
    }
  }

  return pose_size;
}

bool save_tum_poses(const vector<IMUST> &x_buf, const string &filename)
{
  ofstream outFile(filename);
  if(!outFile.is_open())
  {
    printf("open %s fail\n", filename.c_str());
    return false;
  }

  outFile << std::setprecision(19);
  for(const IMUST &pose: x_buf)
  {
    Eigen::Quaterniond q(pose.R);
    q.normalize();
    outFile << pose.t << " "
            << pose.p.x() << " " << pose.p.y() << " " << pose.p.z() << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
  }

  outFile.close();
  printf("Saved %zu optimized poses to %s\n", x_buf.size(), filename.c_str());
  return true;
}

void read_file(vector<IMUST> &x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls, string &prename, int dataset)
{
  if (dataset != 1)
    prename = prename + "/datas/benchmark_realworld/";

  PLV(3) poss; PLM(3) rots;
  vector<double> tims;
  vector<string> pcd_names;
  int pose_size;
  if (dataset == 1) {
    pose_size = read_tum_pose(tims, pcd_names, rots, poss, prename);
  } else {
    pose_size = read_pose(tims, rots, poss, prename);
  }

  if(pose_size == 0) return;
  
  for(int m=0; m<pose_size; m++)
  {
    string filename = prename + "full" + to_string(m) + ".pcd";
    if (dataset == 1) {
      filename = join_path(join_path(prename, "pcd"), pcd_names[m]);
    }
    pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<pcl::PointXYZI> pl_tem;
    if(pcl::io::loadPCDFile(filename, pl_tem) < 0) {
      printf("Failed to load %s\n", filename.c_str());
      continue;
    }
    for(pcl::PointXYZI &pp: pl_tem.points)
    {
      PointType ap;
      ap.x = pp.x; ap.y = pp.y; ap.z = pp.z;
      ap.intensity = pp.intensity;
      pl_ptr->push_back(ap);
    }

    pl_fulls.push_back(pl_ptr);

    IMUST curr;
    curr.R = rots[m]; curr.p = poss[m]; curr.t = tims[m];
    x_buf.push_back(curr);
  }
  

}

void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls)
{
  IMUST es0 = x_buf[0];
  for(uint i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  pcl::PointCloud<PointType> pl_send, pl_path;
  int winsize = x_buf.size();
  for(int i=0; i<winsize; i++)
  {
    pcl::PointCloud<PointType> pl_tem = *pl_fulls[i];
    down_sampling_voxel(pl_tem, 0.05);
    pl_transform(pl_tem, x_buf[i]);
    pl_send += pl_tem;

    if((i%200==0 && i!=0) || i == winsize-1)
    {
      pub_pl_func(pl_send, pub_show);
      pl_send.clear();
      sleep(0.5);
    }

    PointType ap;
    ap.x = x_buf[i].p.x();
    ap.y = x_buf[i].p.y();
    ap.z = x_buf[i].p.z();
    ap.curvature = i;
    pl_path.push_back(ap);
  }

  pub_pl_func(pl_path, pub_path);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "benchmark2");
  ros::NodeHandle n;
  pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
  pub_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
  pub_show = n.advertise<sensor_msgs::PointCloud2>("/map_show", 100);
  pub_cute = n.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);

  string prename, ofname;
  vector<IMUST> x_buf;
  vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;

  n.param<double>("voxel_size", voxel_size, 1);
  string file_path;
  n.param<string>("file_path", file_path, "");
  int dataset;
  n.param<int>("dataset", dataset, 0);

  n.param<int>("max_frames", max_frames, 100);

  read_file(x_buf, pl_fulls, file_path, dataset);
  if(x_buf.empty() || pl_fulls.empty()) {
    printf("No valid input frames were loaded.\n");
    return 1;
  }

  IMUST es0 = x_buf[0];
  for(uint i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  string output_pose_file;
  n.param<string>("output_pose_file", output_pose_file, "");
  if(output_pose_file.empty())
    output_pose_file = join_path(file_path, "optimized_poses.tum");

  win_size = x_buf.size();
  printf("The size of poses: %d\n", win_size);

  data_show(x_buf, pl_fulls);
  printf("Check the point cloud with the initial poses.\n");
  printf("If no problem, input '1' to continue or '0' to exit...\n");
  int a; cin >> a; if(a==0) exit(0);

  pcl::PointCloud<PointType> pl_full, pl_surf, pl_path, pl_send;
  for(int iterCount=0; iterCount<1; iterCount++)
  { 
    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

    eigen_value_array[0] = 1.0 / 16;
    eigen_value_array[1] = 1.0 / 16;
    eigen_value_array[2] = 1.0 / 9;

    for(int i=0; i<win_size; i++)
      cut_voxel(surf_map, *pl_fulls[i], x_buf[i], i);

    pcl::PointCloud<PointType> pl_send;
    pub_pl_func(pl_send, pub_show);

    pcl::PointCloud<PointType> pl_cent; pl_send.clear();
    VOX_HESS voxhess;
    for(auto iter=surf_map.begin(); iter!=surf_map.end() && n.ok(); iter++)
    {
      iter->second->recut(win_size);
      iter->second->tras_opt(voxhess, win_size);
      iter->second->tras_display(pl_send, win_size);
    }

    pub_pl_func(pl_send, pub_cute);
    printf("\nThe planes (point association) cut by adaptive voxelization.\n");
    printf("If the planes are too few, the optimization will be degenerated and fail.\n");
    printf("If no problem, input '1' to continue or '0' to exit...\n");
    int a; cin >> a; if(a==0) exit(0);
    pl_send.clear(); pub_pl_func(pl_send, pub_cute);

    if(voxhess.plvec_voxels.size() < 3 * x_buf.size())
    {
      printf("Initial error too large.\n");
      printf("Please loose plane determination criteria for more planes.\n");
      printf("The optimization is terminated.\n");
      exit(0);
    }

    BALM2 opt_lsv;
    opt_lsv.damping_iter(x_buf, voxhess);

    for(auto iter=surf_map.begin(); iter!=surf_map.end();)
    {
      delete iter->second;
      surf_map.erase(iter++);
    }
    surf_map.clear();

    malloc_trim(0);
  }

  save_tum_poses(x_buf, output_pose_file);

  printf("\nRefined point cloud is publishing...\n");
  malloc_trim(0);
  data_show(x_buf, pl_fulls);
  printf("\nRefined point cloud is published.\n");

  ros::spin();
  return 0;

}
