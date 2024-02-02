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
#include <malloc.h>

#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
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

int read_pose(vector<double> &tims, PLM(3) &rots, PLV(3) &poss, string &readname)
{
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

  while(getline(inFile, lineStr))
  {
    stringstream ss(lineStr);
    while(getline(ss, str, ' '))
      nums.push_back(stod(str));

      for(int j=0; j<16; j++)
        aff(j) = nums[j];

      Eigen::Matrix4d affT = aff.transpose();

      rots.push_back(affT.block<3, 3>(0, 0));
      poss.push_back(affT.block<3, 1>(0, 3));
      tims.push_back(pose_size);
      nums.clear();
      pose_size++;
  }

  // print last pose
  cout << "Last pose: " << endl;
  cout << "R: " << rots[pose_size-1] << endl;
  cout << "p: " << poss[pose_size-1] << endl;

  return pose_size;
}

// copied from https://stackoverflow.com/questions/11140483/how-to-get-list-of-files-with-a-specific-extension-in-a-given-folder
/**
 * \brief   Return the filenames of all files that have the specified extension
 *          in the specified directory and all subdirectories.
 */
std::vector<fs::path> get_all(fs::path const & root, std::string const & ext)
{
    std::vector<fs::path> paths;

    if (fs::exists(root) && fs::is_directory(root))
    {
        for (auto const & entry : fs::recursive_directory_iterator(root))
        {
            if (fs::is_regular_file(entry) && entry.path().extension() == ext)
                paths.emplace_back(entry.path());
        }
    }

    return paths;
}

void read_file(vector<IMUST> &x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls, 
    string &posefilename, string &session1dir, string &session2dir) {
  PLV(3) poss; PLM(3) rots;
  vector<double> tims;
  int pose_size = read_pose(tims, rots, poss, posefilename);
  auto fnlist1 = get_all(session1dir, ".pcd");
  auto fnlist2 = get_all(session2dir, ".pcd");
  auto fnlist = fnlist1;
  fnlist.insert(fnlist.end(), fnlist2.begin(), fnlist2.end());
  if (fnlist.size() != pose_size) {
    cerr << "Warning: Number of poses and point clouds do not match." << endl;
  }

  // only use the last few pose-frame pairs
  size_t lastk = 16;
  size_t skip = fnlist.size() - lastk;
  fnlist = std::vector<fs::path>(fnlist.begin() + skip, fnlist.end());
  tims = std::vector<double>(tims.begin() + skip, tims.end());
  rots = PLM(3)(rots.begin() + skip, rots.end());
  poss = PLV(3)(poss.begin() + skip, poss.end());

  for(int m=0; m<fnlist.size(); m++)
  {
    string filename = fnlist[m].string();

    pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<pcl::PointXYZI> pl_tem;
    pcl::io::loadPCDFile(filename, pl_tem);
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

void save_poses(const vector<IMUST> &x_buf, const string &outposefile) {
  ofstream s(outposefile, ios::out);
  for (const auto &x : x_buf) {
    Eigen::Quaterniond q(x.R);
    s << std::fixed << std::setprecision(10) << x.t << " " << std::fixed << std::setprecision(5)
      << x.p[0] << " " << x.p[1] << " " << x.p[2] << " " << std::setprecision(10)
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  s.close();
  std::cout << "Saved resulting poses to: " << outposefile << std::endl;
}

void left_multiply(const IMUST &es0, vector<IMUST> &x_buf) {
  for(uint i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R * x_buf[i].p + es0.p;
    x_buf[i].R = es0.R * x_buf[i].R;
  }
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
  string session1pcddir;
  n.param<string>("session1pcddir", session1pcddir, "");
  string session2pcddir;
  n.param<string>("session2pcddir", session2pcddir, "");
  string outposefile;
  n.param<string>("outposefile", outposefile, "");
  string refposefile;
  size_t lastindex = outposefile.find_last_of(".");
  if (lastindex != string::npos)
    refposefile = outposefile.substr(0, lastindex) + "orig.txt";

  read_file(x_buf, pl_fulls, file_path, session1pcddir, session2pcddir);
  save_poses(x_buf, refposefile);

  IMUST es0 = x_buf[0];
  for(uint i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

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

  printf("\nRefined point cloud is publishing...\n");
  malloc_trim(0);
  data_show(x_buf, pl_fulls);
  printf("\nRefined point cloud is published.\n");
  left_multiply(es0, x_buf);
  save_poses(x_buf, outposefile);
  ros::spin();
  return 0;
}

