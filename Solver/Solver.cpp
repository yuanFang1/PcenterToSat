#include "Solver.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>

#include <cmath>


using namespace std;


namespace szx {

#pragma region Solver::Cli
	int Solver::Cli::run(int argc, char * argv[]) {
		Log(LogSwitch::Szx::Cli) << "parse command line arguments." << endl;
		Set<String> switchSet;
		Map<String, char*> optionMap({ // use string as key to compare string contents instead of pointers.
			{ InstancePathOption(), nullptr },
			{ SolutionPathOption(), nullptr },
			{ RandSeedOption(), nullptr },
			{ TimeoutOption(), nullptr },
			{ MaxIterOption(), nullptr },
			{ JobNumOption(), nullptr },
			{ RunIdOption(), nullptr },
			{ EnvironmentPathOption(), nullptr },
			{ ConfigPathOption(), nullptr },
			{ LogPathOption(), nullptr }
			});

		for (int i = 1; i < argc; ++i) { // skip executable name.
			auto mapIter = optionMap.find(argv[i]);
			if (mapIter != optionMap.end()) { // option argument.
				mapIter->second = argv[++i];
			}
			else { // switch argument.
				switchSet.insert(argv[i]);
			}
		}

		Log(LogSwitch::Szx::Cli) << "execute commands." << endl;
		if (switchSet.find(HelpSwitch()) != switchSet.end()) {
			cout << HelpInfo() << endl;
		}

		if (switchSet.find(AuthorNameSwitch()) != switchSet.end()) {
			cout << AuthorName() << endl;
		}

		Solver::Environment env;
		env.load(optionMap);
		if (env.instPath.empty() || env.slnPath.empty()) { return -1; }

		Solver::Configuration cfg;
		cfg.load(env.cfgPath);

		Log(LogSwitch::Szx::Input) << "load instance " << env.instPath << " (seed=" << env.randSeed << ")." << endl;
		Problem::Input input;
		if (!input.load(env.instPath)) { return -1; }

		Solver solver(input, env, cfg);
		solver.solve();

		pb::Submission submission;
		submission.set_thread(to_string(env.jobNum));
		submission.set_instance(env.friendlyInstName());
		submission.set_duration(to_string(solver.timer.elapsedSeconds()) + "s");

		solver.output.save(env.slnPath, submission);
#if SZX_DEBUG
		solver.output.save(env.solutionPathWithTime(), submission);
		solver.record();
#endif // SZX_DEBUG

		return 0;
	}
#pragma endregion Solver::Cli

#pragma region Solver::Environment
	void Solver::Environment::load(const Map<String, char*> &optionMap) {
		char *str;

		str = optionMap.at(Cli::EnvironmentPathOption());
		if (str != nullptr) { loadWithoutCalibrate(str); }

		str = optionMap.at(Cli::InstancePathOption());
		if (str != nullptr) { instPath = str; }

		str = optionMap.at(Cli::SolutionPathOption());
		if (str != nullptr) { slnPath = str; }

		str = optionMap.at(Cli::RandSeedOption());
		if (str != nullptr) { randSeed = atoi(str); }

		str = optionMap.at(Cli::TimeoutOption());
		if (str != nullptr) { msTimeout = static_cast<Duration>(atof(str) * Timer::MillisecondsPerSecond); }

		str = optionMap.at(Cli::MaxIterOption());
		if (str != nullptr) { maxIter = atoi(str); }

		str = optionMap.at(Cli::JobNumOption());
		if (str != nullptr) { jobNum = atoi(str); }

		str = optionMap.at(Cli::RunIdOption());
		if (str != nullptr) { rid = str; }

		str = optionMap.at(Cli::ConfigPathOption());
		if (str != nullptr) { cfgPath = str; }

		str = optionMap.at(Cli::LogPathOption());
		if (str != nullptr) { logPath = str; }

		calibrate();
	}

	void Solver::Environment::load(const String &filePath) {
		loadWithoutCalibrate(filePath);
		calibrate();
	}

	void Solver::Environment::loadWithoutCalibrate(const String &filePath) {
		// EXTEND[szx][8]: load environment from file.
		// EXTEND[szx][8]: check file existence first.
	}

	void Solver::Environment::save(const String &filePath) const {
		// EXTEND[szx][8]: save environment to file.
	}
	void Solver::Environment::calibrate() {
		// adjust thread number.
		int threadNum = thread::hardware_concurrency();
		if ((jobNum <= 0) || (jobNum > threadNum)) { jobNum = threadNum; }

		// adjust timeout.
		msTimeout -= Environment::SaveSolutionTimeInMillisecond;
	}
#pragma endregion Solver::Environment

#pragma region Solver::Configuration
	void Solver::Configuration::load(const String &filePath) {
		// EXTEND[szx][5]: load configuration from file.
		// EXTEND[szx][8]: check file existence first.
	}

	void Solver::Configuration::save(const String &filePath) const {
		// EXTEND[szx][5]: save configuration to file.
	}
#pragma endregion Solver::Configuration

#pragma region Solver
	bool Solver::solve() {
		init();

		int workerNum = (max)(1, env.jobNum / cfg.threadNumPerWorker);
		cfg.threadNumPerWorker = env.jobNum / workerNum;
		List<Solution> solutions(workerNum, Solution(this));
		List<bool> success(workerNum);

		Log(LogSwitch::Szx::Framework) << "launch " << workerNum << " workers." << endl;
		List<thread> threadList;
		threadList.reserve(workerNum);
		for (int i = 0; i < workerNum; ++i) {
			// TODO[szx][2]: as *this is captured by ref, the solver should support concurrency itself, i.e., data members should be read-only or independent for each worker.
			// OPTIMIZE[szx][3]: add a list to specify a series of algorithm to be used by each threads in sequence.
			threadList.emplace_back([&, i]() { success[i] = optimize(solutions[i], i); });
		}
		for (int i = 0; i < workerNum; ++i) { threadList.at(i).join(); }

		Log(LogSwitch::Szx::Framework) << "collect best result among all workers." << endl;
		int bestIndex = -1;
		Length bestValue = Problem::MaxDistance;
		for (int i = 0; i < workerNum; ++i) {
			if (!success[i]) { continue; }
			Log(LogSwitch::Szx::Framework) << "worker " << i << " got " << solutions[i].coverRadius << endl;
			if (solutions[i].coverRadius >= bestValue) { continue; }
			bestIndex = i;
			bestValue = solutions[i].coverRadius;
		}

		env.rid = to_string(bestIndex);
		if (bestIndex < 0) { return false; }
		output = solutions[bestIndex];
		return true;
	}

	void Solver::record() const {
#if SZX_DEBUG
		int generation = 0;

		ostringstream log;

		System::MemoryUsage mu = System::peakMemoryUsage();

		Length obj = output.coverRadius;
		Length checkerObj = -1;
		bool feasible = 1;
		feasible = check(checkerObj);

		// record basic information.
		log << env.friendlyLocalTime() << ","
			<< env.rid << ","
			<< env.instPath << ","
			<< feasible << "," << (obj - checkerObj) << ",";
		if (Problem::isTopologicalGraph(input)) {
			log << obj << ",";
		}
		else {
			auto oldPrecision = log.precision();
			log.precision(2);
			log << fixed << setprecision(2) << (obj / aux.objScale) << ",";
			log.precision(oldPrecision);
		}
		log << timer.elapsedSeconds() << ","
			<< mu.physicalMemory << "," << mu.virtualMemory << ","
			<< env.randSeed << ","
			<< cfg.toBriefStr() << ","
			<< generation << "," << iteration << ",";

		// record solution vector.
		for (auto c = output.centers().begin(); c != output.centers().end(); ++c) {
			log << *c << " ";
		}
		log << endl;

		// append all text atomically.
		static mutex logFileMutex;
		lock_guard<mutex> logFileGuard(logFileMutex);

		ofstream logFile(env.logPath, ios::app);
		logFile.seekp(0, ios::end);
		if (logFile.tellp() <= 0) {
			logFile << "Time,ID,Instance,Feasible,ObjMatch,Distance,Duration,PhysMem,VirtMem,RandSeed,Config,Generation,Iteration,Solution" << endl;
		}
		logFile << log.str();
		logFile.close();
#endif // SZX_DEBUG
	}

	bool Solver::check(Length &checkerObj) const {
#if SZX_DEBUG
		enum CheckerFlag {
			IoError = 0x0,
			FormatError = 0x1,
			TooManyCentersError = 0x2
		};

		checkerObj = System::exec("Checker.exe " + env.instPath + " " + env.solutionPathWithTime());
		if (checkerObj > 0) { return true; }
		checkerObj = ~checkerObj;
		if (checkerObj == CheckerFlag::IoError) { Log(LogSwitch::Checker) << "IoError." << endl; }
		if (checkerObj & CheckerFlag::FormatError) { Log(LogSwitch::Checker) << "FormatError." << endl; }
		if (checkerObj & CheckerFlag::TooManyCentersError) { Log(LogSwitch::Checker) << "TooManyCentersError." << endl; }
		return false;
#else
		checkerObj = 0;
		return true;
#endif // SZX_DEBUG
	}

	void Solver::init() {
		ID nodeNum = input.graph().nodenum();

		aux.adjMat.init(nodeNum, nodeNum);
		fill(aux.adjMat.begin(), aux.adjMat.end(), Problem::MaxDistance);
		for (ID n = 0; n < nodeNum; ++n) { aux.adjMat.at(n, n) = 0; }

		if (Problem::isTopologicalGraph(input)) {
			aux.objScale = Problem::TopologicalGraphObjScale;
			for (auto e = input.graph().edges().begin(); e != input.graph().edges().end(); ++e) {
				// only record the last appearance of each edge.
				aux.adjMat.at(e->source(), e->target()) = e->length();
				aux.adjMat.at(e->target(), e->source()) = e->length();
			}

			Timer timer(30s);
			constexpr bool IsUndirectedGraph = true;
			IsUndirectedGraph
				? Floyd::findAllPairsPaths_symmetric(aux.adjMat)
				: Floyd::findAllPairsPaths_asymmetric(aux.adjMat);
			Log(LogSwitch::Preprocess) << "Floyd takes " << timer.elapsedSeconds() << " seconds." << endl;
		}
		else { // geometrical graph.
			aux.objScale = Problem::GeometricalGraphObjScale;
			for (ID n = 0; n < nodeNum; ++n) {
				double nx = input.graph().nodes(n).x();
				double ny = input.graph().nodes(n).y();
				for (ID m = 0; m < nodeNum; ++m) {
					if (n == m) { continue; }
					aux.adjMat.at(n, m) = lround(aux.objScale * hypot(
						nx - input.graph().nodes(m).x(), ny - input.graph().nodes(m).y()));
				}
			}
		}

		aux.coverRadii.init(nodeNum);
		fill(aux.coverRadii.begin(), aux.coverRadii.end(), Problem::MaxDistance);
	}
	struct SetCoverNodes {
	public:
		ID set_id;
		vector<ID> nodes;
	};
	bool coverSets_sort(const SetCoverNodes &c1, const SetCoverNodes &c2) {
		return c1.nodes.size() < c2.nodes.size();
	}
	///数据预处理相关代码
	bool ifCanBeDominated(vector<ID> &N3_vw, ID u, vector<vector<bool>> &belongToNeigh) {/*判断结点u能否支配某个结点集*/
		for (auto iter = N3_vw.begin(); iter != N3_vw.end(); ++iter) {
			ID t = *iter;
			if (u != t && !belongToNeigh[u][t]) {/*说明t不属于u的邻居结点，即u无法支配t */
				return false;
			}
		}
		return true;
	}
	void removeNodesSet(map<ID, map<ID, set<ID>>> &N, map<ID, map<ID, set<ID>>> &N1) {
		for (auto N_iter_v = N.begin(), N1_iter_v = N1.begin(); N_iter_v != N.end(); ++N_iter_v, ++N1_iter_v) {
			ID v = (*N_iter_v).first;
			map<ID, set<ID>> &tempN = (*N_iter_v).second;
			map<ID, set<ID>> &tempN1 = (*N1_iter_v).second;
			for (auto N_iter_w = tempN.begin(), N1_iter_w = tempN1.begin(); N_iter_w != tempN.end(); ++N_iter_w, ++N1_iter_w) {
				ID w = (*N_iter_w).first;
				set<ID> &N_vw = (*N_iter_w).second;
				set<ID> &N1_vw = (*N1_iter_w).second;
				set<ID> newSet;
				if (N_vw.size() == N1_vw.size()) {/* 将N1(v,w)中的元素从N(v,w)中去除 */
					N_vw.clear();
				}
				else {
					for (auto iter = N_vw.begin(); iter != N_vw.end(); ++iter) {
						ID u = *iter;
						auto find = N1_vw.find(u);
						if (find == N1_vw.end()) {/* 说明N1(v,w)中不含u */
							newSet.insert(u);
						}
					}
					N_vw = newSet;
				}
			}
		}
	}
	void eraseElem(std::vector<ID> &list, ID elem) {
		bool find = false;
		for (auto j = 0; j < list.size(); ++j) {
			if (list[j] == elem) {
				list.erase(list.begin() + j);
				find = true;
				break;
			}
		}
		if (!find) {
			cout << "erase error\n";
			exit(1);
		}
	}
	void floyd(std::vector<std::vector<Length>> &adjMat) {
		int size = adjMat.size();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < size; k++) {
					Length select = (adjMat[j][i] == INT_MAX || adjMat[i][k] == INT_MAX)
						? INT_MAX : (adjMat[j][i] + adjMat[i][k]);

					if (adjMat[j][k] > select) {
						adjMat[j][k] = select;
					}
				}
			}
		}
	}
	void handleWhiteNodes(int size, vector<bool> &isWhiteNode, vector<vector<ID>> &new_neighbours, std::set<ID> &trueSets, std::set<ID> &delSets) {
		/*R1 :delete edges between white vertices*/
		for (auto i = 0; i < size; ++i) {
			if (isWhiteNode[i]) {/* 只针对已经被支配的结点 */
				for (auto j = 0; j < new_neighbours[i].size(); ++j) {
					if (isWhiteNode[new_neighbours[i][j]]) {
						new_neighbours[i].erase(new_neighbours[i].begin() + j);
						j--;
					}
				}
			}
		}
		/* 删除掉度为0或1的被支配结点 */
		for (auto i = 0; i < size; ++i) {
			if (isWhiteNode[i]) {/* 已经被支配的结点的邻居结点全部都是未被支配的结点 */
				auto iter1 = trueSets.find(i);
				auto iter2 = delSets.find(i);
				if (iter1 == trueSets.end() && iter2 == delSets.end()) {/* 说明是没被remove的结点 */
					if (new_neighbours[i].size() == 0) {
						delSets.insert(i);
					}
					else if (new_neighbours[i].size() == 1) {
						ID node = new_neighbours[i][0];
						eraseElem(new_neighbours[node], i);
						delSets.insert(i);
						new_neighbours[i].clear();
					}
				}
			}
		}
		/* 删除掉度为2且满足一定条件的被支配结点 */
		for (auto i = 0; i < size; ++i) {
			if (isWhiteNode[i]) {
				auto iter1 = trueSets.find(i);
				auto iter2 = delSets.find(i);
				if (iter1 == trueSets.end() && iter2 == delSets.end()) {/* 说明是没被remove的结点 */
					if (new_neighbours[i].size() == 2) {
						ID node1 = new_neighbours[i][0];
						ID node2 = new_neighbours[i][1];
						bool notFind = true;
						for (auto j = 0; j < new_neighbours[node1].size() && notFind; ++j) {
							ID n1 = new_neighbours[node1][j];
							if (n1 == node2) {
								notFind = false;
								break;
							}
						}
						for (auto j = 0; j < new_neighbours[node1].size() && notFind; ++j) {
							ID n1 = new_neighbours[node1][j];
							for (auto k = 0; k < new_neighbours[node2].size() && notFind; ++k) {
								ID n2 = new_neighbours[node2][k];
								if (n1 == n2) {
									notFind = false;
									break;
								}
							}
						}
						if (!notFind) {
							eraseElem(new_neighbours[node1], i);
							eraseElem(new_neighbours[node2], i);
							delSets.insert(i);
						}
					}
				}
			}
		}

		/* 删除掉度为3且满足一定条件的被支配结点 */
		for (auto i = 0; i < size; ++i) {
			if (isWhiteNode[i]) {
				auto iter1 = trueSets.find(i);
				auto iter2 = delSets.find(i);
				if (iter1 == trueSets.end() && iter2 == delSets.end()) {/* 说明是没被remove的结点 */
					if (new_neighbours[i].size() == 3) {
						bool find = false;
						for (auto j = 0; j < new_neighbours[i].size(); ++j) {
							ID n1, n2;
							ID node = new_neighbours[i][j];
							if (j == 0) {
								n1 = new_neighbours[i][1];
								n2 = new_neighbours[i][2];
							}
							else if (j == 1) {
								n1 = new_neighbours[i][0];
								n2 = new_neighbours[i][2];
							}
							else if (j == 2) {
								n1 = new_neighbours[i][0];
								n2 = new_neighbours[i][1];
							}
							bool notFind1 = true, notFind2 = true;
							for (auto k = 0; k < new_neighbours[node].size(); ++k) {
								if (n1 == new_neighbours[node][k]) {
									notFind1 = false;
								}
								else if (n2 == new_neighbours[node][k]) {
									notFind2 = false;
								}

							}
							if (!notFind1 && !notFind2) {
								find = true;
								break;
							}
						}
						if (find) {
							for (auto j = 0; j < new_neighbours[i].size(); ++j) {
								ID node = new_neighbours[i][j];
								eraseElem(new_neighbours[node], i);
							}
							delSets.insert(i);
						}
					}
				}
			}
		}
	}
	void dataReductionRule1(std::vector<std::vector<ID>> &neighbours,std::set<ID> &trueSets, std::set<ID> &delSets) {
		vector<vector<bool>> belongTo; /* belongTo[i][j]:结点j是否属于结点i能覆盖的点 */
		vector<vector<ID>> N1;
		vector<vector<ID>> N2;
		vector<vector<ID>> N3;

		vector<vector<bool>> belongToN1;
		vector<vector<bool>> belongToN2;


		int size = neighbours.size();
		belongTo.resize(size);
		N1.resize(size);
		N2.resize(size);
		N3.resize(size);
		belongToN1.resize(size);
		belongToN2.resize(size);

		for (auto i = 0; i < size; ++i) {
			belongTo[i].resize(size);
			belongToN1[i].resize(size);
			belongToN2[i].resize(size);
		}
		for (auto v = 0; v < size; ++v) {
			for (auto j = 0; j < neighbours[v].size(); ++j) {
				ID u = neighbours[v][j];
				belongTo[v][u] = true;
			}
		}

		for (auto v = 0; v < neighbours.size(); ++v) {
			for (auto j = 0; j < neighbours[v].size(); ++j) {
				ID u = neighbours[v][j]; /* u为v的邻居结点 */
				for (auto k = 0; k < neighbours[u].size(); ++k) {
					ID t = neighbours[u][k];/* t为u的邻居结点 */
					if (!belongTo[v][t] && t != v) { /* v的邻居结点存在有不是v的邻居结点的，说明结点u属于N1 */
						N1[v].push_back(u);
						belongToN1[v][u] = true; /* u属于N1(v) */
						break;
					}
				}

			}
		}
		for (auto v = 0; v < size; ++v) {
			if (neighbours[v].size() != N1[v].size()) {
				for (auto i = 0; i < neighbours[v].size(); ++i) {
					ID u = neighbours[v][i];
					if (!belongToN1[v][u]) {
						for (auto j = 0; j < neighbours[u].size(); ++j) {
							ID t = neighbours[u][j];
							if (belongToN1[v][t]) {
								N2[v].push_back(u);
								belongToN2[v][u] = true;
								break;
							}
						}
					}
				}
			}
		}
		for (auto v = 0; v < size; ++v) {
			if (neighbours[v].size() != (N1[v].size() + N2[v].size())) {
				//cout << "N3_vw: " << neighbours[v].size() - N1[v].size() << endl;
				for (auto i = 0; i < neighbours[v].size(); ++i) {
					ID u = neighbours[v][i];
					if (!belongToN1[v][u] && !belongToN2[v][u]) {/* u不属于N1(v),u不属于N2(v)  */
						N3[v].push_back(u);
						if (neighbours[u].size() != (N1[u].size() + N2[u].size())) {
							/*cout << u << ":";
							for (auto j = 0; j < neighbours[u].size(); ++j) {
								cout << neighbours[u][j] << " ";
							}
							cout << endl;*/
						}
					}
				}
			}
		}
		
		for (auto v = 0; v < size; ++v) {
			auto iter = delSets.find(v);
			if (N3[v].size() > 0 && iter == delSets.end()) {
				trueSets.insert(v);
				//cout << "the N3_vw of " << v << ": ";
				for (auto i = 0; i < N3[v].size(); ++i) {
					//cout << N3[v][i] << "  ";
					delSets.insert(N3[v][i]);
				}
				//cout << endl;

				//cout << "the N2 of " << v << ": ";
				for (auto i = 0; i < N2[v].size(); ++i) {
					//cout << N2[v][i] << "  ";
					delSets.insert(N2[v][i]);
				}
				//cout << endl;
			}

		}
		cout <<"rule1:"<< trueSets.size() << endl;
		cout <<"rule1:"<< delSets.size() << endl;
	}
	void reduction_merge(const std::vector<std::vector<ID>> &neighbours, int v, int w, std::vector<ID> &N_vw) {
		const std::vector<ID> &nv = neighbours[v];
		const std::vector<ID> &nw = neighbours[w];
		auto size_v = nv.size();
		auto size_w = nw.size();
		auto index_v = 0, index_w = 0;
		N_vw.reserve(size_v + size_w);
		while (index_v < size_v &&index_w < size_w) {
			if (nv[index_v] == w) { index_v++; continue; }
			if (nw[index_w] == v) { index_w++; continue; }

			if (nv[index_v] < nw[index_w]) {
				N_vw.push_back(nv[index_v]);
				index_v++;
			}
			else if (nv[index_v] == nw[index_w]) {
				N_vw.push_back(nv[index_v]);
				index_v++;
				index_w++;
			}
			else if (nv[index_v] > nw[index_w]) {
				N_vw.push_back(nw[index_w]);
				index_w++;
			}
		}
		while (index_v < size_v) {
			N_vw.push_back(nv[index_v]);
			index_v++;
		}
		while (index_w < size_w) {
			N_vw.push_back(nw[index_w]);
			index_w++;
		}
	}
	void dataReductionRule2(const std::vector<std::vector<ID>> &neighbours, std::set<std::vector<ID>> &atLeastTrueSets, std::set<ID> &trueSets, std::set<ID> &delSets) {
		int size = neighbours.size();
		vector<vector<bool>> belongToNeigh; /* belongToNeigh[v][u] ，标记结点u是否属于N(v) */
		vector<vector<Length>> adjMat;
		map<ID, map<ID, set<ID>>> N; /* 用于存储N(v,w) */
		map<ID, map<ID, set<ID>>> N1;
		map<ID, map<ID, set<ID>>> N2;
		map<ID, map<ID, set<ID>>> N3;

		///内存申请
		belongToNeigh.resize(size);
		adjMat.resize(size);
		for (auto v = 0; v < size; ++v) {
			belongToNeigh[v].resize(size);
			adjMat[v].resize(size);
		}
		for (auto v = 0; v < size; ++v) {
			for (auto w = 0; w < size; ++w) {
				adjMat[v][w] = INT_MAX;
			}
		}
		for (auto v = 0; v < size; ++v) {
			for (auto i = 0; i < neighbours[v].size(); ++i) {
				ID u = neighbours[v][i];
				belongToNeigh[v][u] = true;
				adjMat[v][u] = 1;
			}
		}
		floyd(adjMat);
		for (auto v = 0; v < size; ++v) {
			auto find_del = delSets.find(v),find_true = trueSets.find(v);
			if (find_del != delSets.end() ) continue;
			for (auto w = v + 1; w < size; ++w) {
				if (adjMat[v][w] <= 3) {
					auto find_del = delSets.find(w), find_true = trueSets.find(w);
					if (find_del != delSets.end()) continue;
					vector<ID> N_vw; //用于保存N(v,w)
					vector<bool> isBelongToN_vw,isBelongToN1_vw,isBelongToN2_vw;
					vector<ID> N1_vw,N2_vw,N3_vw;
					reduction_merge(neighbours, v, w, N_vw);
					if (N_vw.size() > 1) {
						isBelongToN_vw.resize(size);
						isBelongToN1_vw.resize(size);
						isBelongToN2_vw.resize(size);
						N1_vw.reserve(N_vw.size());
						for (auto i = 0; i < N_vw.size(); ++i) {
							isBelongToN_vw[N_vw[i]] = true;
						}
						for (auto i = 0; i < N_vw.size(); ++i) {//获取N1(v,w)
							ID u = N_vw[i];
							for (auto j = 0; j < neighbours[u].size(); ++j) {
								ID t = neighbours[u][j];
								if (t == v || t == w)continue;
								if (!isBelongToN_vw[t]) {
									N1_vw.push_back(u);
									isBelongToN1_vw[u] = true;
									break;
								}
							}
						}
						if (N_vw.size() != N1_vw.size()) {
							N2_vw.reserve(N_vw.size() - N1_vw.size());
							for (auto i = 0; i < N_vw.size(); ++i) {//获取N2(v,w)
								ID u = N_vw[i];
								if (!isBelongToN1_vw[u]) {/* 遍历N(v,w)中除去N1(v,w)的结点u */
									for (auto j = 0; j < neighbours[u].size(); ++j) {
										ID t = neighbours[u][j];
										if (isBelongToN1_vw[t]) {
											N2_vw.push_back(u);
											isBelongToN2_vw[u] = true;
											break;
										}
									}
								}
							}
							if (N_vw.size() - (N1_vw.size() + N2_vw.size()) > 1) {// |N3(v,w)|需要大于1
								N3_vw.reserve(N_vw.size() - (N1_vw.size() + N2_vw.size()));
								for (auto i = 0; i < N_vw.size(); ++i) {//获取N3(v,w)
									ID u = N_vw[i];
									if (!isBelongToN1_vw[u] && !isBelongToN2_vw[u]) {
										N3_vw.push_back(u);
									}
								}

								//接下来判断结点是否可以被确定为中心
								bool dominate = false;
								vector<ID> trueSet;
								for (auto iter = N3_vw.begin(); iter != N3_vw.end(); ++iter) {/*判断N3(v,w)中的某个点u能否支配整个N3(v,w)*/
									ID u = *iter;
									if (ifCanBeDominated(N3_vw, u, belongToNeigh)) {
										dominate = true;
										break;
									}
								}
								if (!dominate) {
									for (auto iter = N2_vw.begin(); iter != N2_vw.end(); ++iter) {/*判断N2(v,w)中的某个点u能否支配整个N3(v,w)*/
										ID u = *iter;
										if (ifCanBeDominated(N3_vw, u, belongToNeigh)) {
											dominate = true;
											break;
										}
									}
								}
								if (!dominate) {/* 说明N3(v,w)无法被N2(v,w)或N3(v,w)中的某个结点支配 */
									bool dominate_v = ifCanBeDominated(N3_vw, v, belongToNeigh);
									bool dominate_w = ifCanBeDominated(N3_vw, w, belongToNeigh);
									if (dominate_v || dominate_w) {// case1: N3(v,w)能被v或者w支配
										if (dominate_v && dominate_w) {
											auto find_v = trueSets.find(v);
											auto find_w = trueSets.find(w);
											if (find_v == trueSets.end() && find_w == trueSets.end()) {/* 说明v和w都没有被确定为中心 */
												trueSet.push_back(v);
												trueSet.push_back(w);
												atLeastTrueSets.insert(trueSet);
											}
											for (auto iter = N3_vw.begin(); iter != N3_vw.end(); ++iter) {
												ID u = *iter;
												delSets.insert(u);
											}
											for (auto iter = N2_vw.begin(); iter != N2_vw.end(); ++iter) {
												ID u = *iter;
												if (belongToNeigh[v][u] && belongToNeigh[w][u]) {
													delSets.insert(u);
												}
											}
										}
										else if (dominate_v && !dominate_w) {
											trueSets.insert(v);
											for (auto iter = N3_vw.begin(); iter != N3_vw.end(); ++iter) {
												ID u = *iter;
												delSets.insert(u);
											}
											for (auto iter = N2_vw.begin(); iter != N2_vw.end(); ++iter) {
												ID u = *iter;
												if (belongToNeigh[v][u]) {
													delSets.insert(u);
												}
											}
										}
										else if (!dominate_v &&dominate_w) {
											trueSets.insert(w);
											for (auto iter = N3_vw.begin(); iter != N3_vw.end(); ++iter) {
												ID u = *iter;
												delSets.insert(u);
											}
											for (auto iter = N2_vw.begin(); iter != N2_vw.end(); ++iter) {
												ID u = *iter;
												if (belongToNeigh[w][u]) {
													delSets.insert(u);
												}
											}
										}
									}
									else {
										trueSets.insert(v);
										trueSets.insert(w);

										for (auto iter = N3_vw.begin(); iter != N3_vw.end(); ++iter) {
											ID u = *iter;
											delSets.insert(u);
										}
										for (auto iter = N2_vw.begin(); iter != N2_vw.end(); ++iter) {
											ID u = *iter;
											delSets.insert(u);
										}
									}
								}

							}

						}
					}
					
					
				}
			}
		}

		cout << "rule2:" << trueSets.size() << endl;
		cout << "rule2:" << delSets.size() << endl;
		/*for (auto i = 0; i < trueSets.size(); ++i) {
			for (auto j = 0; j < trueSets[i].size(); ++j) {
				cout << trueSets[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
		for (auto i = 0; i < delSets.size(); ++i) {
			cout << "-" << delSets[i] << endl;
		}*/
	}
	
	void dataReductionAdditionRule(std::vector<std::vector<ID>> &neighbours, std::set<std::vector<ID>> &atLeastTrueSets, std::set<ID> &trueSets, std::set<ID> &delSets) {
		vector<bool> isWhiteNode; /* 是否是已经被支配的结点 */
		vector<vector<ID>> new_neighbours;
		int size = neighbours.size();
		isWhiteNode.resize(size);
		new_neighbours.resize(size);
		for (auto i = 0; i < size; ++i) {
			auto iter1 = trueSets.find(i);
			auto iter2 = delSets.find(i);
			if (iter1 != trueSets.end()) {/* 说明结点i已经被确定为中心 */
				for (auto j = 0; j < neighbours[i].size(); ++j) {
					ID node = neighbours[i][j];
					isWhiteNode[node] = true;
				}
			}
			if (iter1 == trueSets.end() && iter2 == delSets.end()) {/* 说明是没被remove的结点 */
				for (auto j = 0; j < neighbours[i].size(); ++j) {
					ID node = neighbours[i][j];
					auto iter1 = trueSets.find(node);
					auto iter2 = delSets.find(node);
					if (iter1 == trueSets.end() && iter2 == delSets.end()) {/*说明node既不在trueSets中也不在delSets中*/
						new_neighbours[i].push_back(node);
					}
				}
			}
		}
		handleWhiteNodes(size, isWhiteNode, new_neighbours, trueSets, delSets);
		/* 处理度为1的未被支配的结点*/
		bool find = false;
		for (auto i = 0; i < size; ++i) {
			if (!isWhiteNode[i]) {
				auto iter1 = trueSets.find(i);
				auto iter2 = delSets.find(i);
				if (iter1 == trueSets.end() && iter2 == delSets.end()) {/* 说明是没被remove的结点 */
					if (new_neighbours[i].size() == 1) {
						delSets.insert(i);
						ID node = new_neighbours[i][0];
						trueSets.insert(node);
						cout << "deg1:" << new_neighbours[node].size() << endl;
						for (auto j = 0; j < new_neighbours[node].size(); ++j) {/* 将node从node的邻居结点的邻域中移除*/
							ID neigh = new_neighbours[node][j];
							if (neigh!= i) {
								find = true;
							}
							isWhiteNode[neigh] = true;
							eraseElem(new_neighbours[neigh], node);
						}
						new_neighbours[node].clear();
					}
				}
			}
		}
		if(find)
			handleWhiteNodes(size, isWhiteNode, new_neighbours, trueSets, delSets);

		/* 处理度为0的未被支配的结点 */
		for (auto i = 0; i < size; ++i) {
			if (!isWhiteNode[i]) {
				auto iter1 = trueSets.find(i);
				auto iter2 = delSets.find(i);
				if (iter1 == trueSets.end() && iter2 == delSets.end()) {/* 说明是没被remove的结点 */
					if (new_neighbours[i].size() == 0) {
						trueSets.insert(i);
					}
				}
			}
		}

		for (auto iter = atLeastTrueSets.begin(); iter != atLeastTrueSets.end(); iter++) {
			bool isTrue = false;
			for (auto i = 0; i < (*iter).size(); ++i) {
				auto m = trueSets.find((*iter)[i]);
				if (m != trueSets.end()) {
					isTrue = true;
					break;
				}
			}
			if (isTrue) {
				atLeastTrueSets.erase(iter);
			}
		}
		
	}
	/**
	* \brief 实现数据预处理
	* \param neighbours :传入参数，邻接表； old_atleastTrueSets:传入传出参数,old_atleastTrueSets[i]包含若干个点，至少有一个点需要成为中心
	* trueSets: 传入传出参数，确定是中心的结点  delSets:传入传出参数，确定不是中心的结点
	*/
	void dataReduction(std::vector<std::vector<ID>> &neighbours, std::set<std::vector<ID>> &old_atLeastTrueSets, std::set<ID> &trueSets, std::set<ID> &delSets) {
		
		dataReductionRule1(neighbours, trueSets, delSets);
		dataReductionRule2(neighbours, old_atLeastTrueSets, trueSets, delSets);

		dataReductionAdditionRule(neighbours, old_atLeastTrueSets, trueSets, delSets);
	}
	
	void getBelongTo(vector<SetCoverNodes> &coverSets, vector<vector<ID>> &belongTo,set<ID> &trueSets,set<ID> &delSets,ZeroBasedConsecutiveIdMap<ID,ID> &residualNodesMap) {
		int old_nodeNum = coverSets.size();
		belongTo.clear();
		ID cnt = 0;
		ofstream ofs;
		ofs.open("old_belong.txt", ios::out);
		if (!ofs.is_open()) {
			cout << "[Fatal] Invalid answer path!" << endl;
		}
		else {
			belongTo.resize(coverSets.size());
			for (auto i = 0; i < coverSets.size(); ++i) {
				for (auto j = 0; j < coverSets[i].nodes.size(); ++j) {
					belongTo[coverSets[i].nodes[j]].push_back(coverSets[i].set_id);
				}
			}
			for (auto i = 0; i < belongTo.size(); ++i) {
				for (auto j = 0; j < belongTo[i].size(); ++j) {
					ofs << belongTo[i][j] << " ";
				}
				ofs << endl;
			}
			
		}
		ofs.close();
		vector<vector<ID>> new_belongTo;
		new_belongTo.resize(coverSets.size());
		for (auto i = 0; i < belongTo.size(); ++i) {
			auto iter1 = trueSets.find(i);
			auto iter2 = delSets.find(i);
			if (iter1 == trueSets.end() && iter2 == delSets.end()) {/* 说明结点i并没有被删除也没有被确定为中心 */
				for (auto j = 0; j < belongTo[i].size(); ++j) {
					ID node = belongTo[i][j];
					auto iter1 = trueSets.find(node);
					auto iter2 = delSets.find(node);
					if (iter1 != trueSets.end()) {/* 要选的结点里node被确定为中心 */
						new_belongTo[i].clear();
						break;
					}
					else if (iter2 == delSets.end()) {/* node没有被删除 */
						new_belongTo[i].push_back(node);
					}
				}
			}
		}


		ofs.open("new_belong.txt", ios::out);
		if (!ofs.is_open()) {
			cout << "[Fatal] Invalid answer path!" << endl;
		}
		else {
			for (auto i = 0; i < new_belongTo.size(); ++i) {
				for (auto j = 0; j < new_belongTo[i].size(); ++j) {
					ofs << new_belongTo[i][j] << " ";
				}
				ofs << endl;
			}

		}
		ofs.close();
		for (auto i = 0; i < old_nodeNum; ++i) {/* 将没有确定的结点通过map映射 */
			auto iter1 = trueSets.find(i);
			auto iter2 = delSets.find(i);
			if (iter1 == trueSets.end() && iter2 == delSets.end()) {/* 说明结点i并没有被删除也没有被确定为中心 */
				residualNodesMap.toConsecutiveId(i);
				cnt++;
			}
		}
		vector<vector<ID>> temp_belongTo;
		temp_belongTo.resize(residualNodesMap.count + 1);
		for (auto i = 0; i < new_belongTo.size(); ++i) {
			for (auto j = 0; j < new_belongTo[i].size(); ++j) {
				temp_belongTo[residualNodesMap.toConsecutiveId(i)].push_back(residualNodesMap.toConsecutiveId(new_belongTo[i][j]));
			}
		}


		belongTo.clear();
		belongTo.resize(residualNodesMap.count+1);
		for (auto i = 0; i < coverSets.size(); ++i) {
			ID id = coverSets[i].set_id;
			auto iter1 = trueSets.find(id);
			auto iter2 = delSets.find(id);
			if (iter1 == trueSets.end() && iter2 == delSets.end()) {/* 说明结点id并没有被删除也没有被确定为中心 */
				for (auto j = 0; j < coverSets[i].nodes.size(); ++j) {
					ID node = coverSets[i].nodes[j];
					auto iter1 = trueSets.find(node);
					auto iter2 = delSets.find(node);
					if (iter1 == trueSets.end() && iter2 == delSets.end()) {
						belongTo[residualNodesMap.toConsecutiveId(node)].push_back(residualNodesMap.toConsecutiveId(id));
					}
				}
			}
		}
		for (auto iter = trueSets.begin(); iter != trueSets.end(); ++iter) {
			ID center = *iter;
			for (auto i = 0; i < coverSets[center].nodes.size(); ++i) {
				ID coverNode = coverSets[center].nodes[i];
				if (residualNodesMap.isArbitraryIdExist(coverNode)) {
					belongTo[residualNodesMap.toConsecutiveId(coverNode)].clear();
				}
			}
		}

		if (belongTo.size() != temp_belongTo.size()) {
			cout << "size error\n";
			exit(1);
		}
		for (auto i = 0; i < belongTo.size(); ++i) {
			if (belongTo[i].size() != temp_belongTo[i].size()) {
				cout << "bel size error\n";
				exit(2);
			}
			for (auto j = 0; j < belongTo[i].size(); ++j) {
				if (belongTo[i][j] != temp_belongTo[i][j]) {
					cout << "not equal\n";
					exit(3);
				}
			}
		}
		
		/*for (auto iter = trueSets.begin(); iter != trueSets.end(); ++iter) {
			ID center = *iter;
			
		}*/
	}
	/*void getBelongTo(vector<SetCoverNodes> &coverSets, vector<vector<ID>> &belongTo, int nodeNum) {
		belongTo.clear();
		belongTo.resize(nodeNum);
		for (auto i = 0; i < coverSets.size(); ++i) {
			for (auto j = 0; j < coverSets[i].nodes.size(); ++j) {
				belongTo[coverSets[i].nodes[j]].push_back(coverSets[i].set_id);
			}
		}
	}*/

	bool Solver::optimize(Solution &sln, ID workerId) {
		Log(LogSwitch::Szx::Framework) << "worker " << workerId << " starts." << endl;

		ID nodeNum = input.graph().nodenum();
		ID centerNum = input.centernum();

		// reset solution state.
		bool status = true;
		auto &centers(*sln.mutable_centers());
		centers.Resize(centerNum, Problem::InvalidId);
		char readLine[100000], tempLine[100000];
		const char *delim = " ";
		char *p;
		Length Radius;
		ifstream param("param.txt");
		cout << env.instPath.substr(9, env.instPath.length() - 14) << endl;
		while (param.getline(readLine, 10000)) {
			string temp(readLine);
			p = strtok(readLine, delim);
			if (temp.find(env.instPath.substr(9, env.instPath.length() - 14)) != string::npos) {
				p = strtok(NULL, delim);
				Radius = atoi(p);
				break;
			}
		}/*从文件中读取已知的最优值*/

		set<int> radius_toBeUse;
		for (auto i = 0; i < nodeNum; ++i) {
			for (auto j = 0; j < nodeNum; ++j) {
				if (aux.adjMat.at(i, j) <= Radius) {
					radius_toBeUse.insert(aux.adjMat.at(i, j));
				}
			}
		}/*将所有小于最优值的边记录下来*/

		sln.coverRadius = 0; // record obj.
		auto it = radius_toBeUse.end();
		it--;
		//for (; it != radius_toBeUse.begin(); --it) {
		Radius = *it;
		cout << "------------------------------------search radius = " << Radius << "--------------------------------------" << endl;
		vector<SetCoverNodes> coverSets; /* coverSets[i]:结点coverSets[i].set_id在指定服务半径下可覆盖的结点集 */
		vector<vector<ID>> neighbours; /* neighbours[i]: 结点i在指定服务半径下可覆盖的结点集*/
		set<ID> delSets;
		set<ID> trueSets;
		set<vector<ID>> old_atLeastTrueSets;
		set<vector<ID>> new_atLeastTrueSets;
		vector<ID> residualNodes;
		vector<vector<ID>> belongTo; /* belongTo[i]:结点i被哪些结点集覆盖 */
		ZeroBasedConsecutiveIdMap<ID, ID> residualNodesMap;

		coverSets.resize(nodeNum);
		neighbours.resize(nodeNum);
		for (auto i = 0; i < nodeNum; ++i) {
			SetCoverNodes scns;
			scns.set_id = i;
			for (auto j = 0; j < nodeNum; ++j) {
				if (aux.adjMat.at(i, j) <= Radius) {
					scns.nodes.push_back(j);
					if (j != i)
						neighbours[i].push_back(j);
				}
			}
			coverSets[i] = scns;
		}
		ofstream ofs;
		ofs.open("reduction.txt", ios::app);
		if (!ofs.is_open()) {
			cout << "[Fatal] Invalid answer path!" << endl;
		}
		//sort(coverSets.begin(), coverSets.end(), coverSets_sort);
		dataReduction(neighbours, old_atLeastTrueSets, trueSets, delSets);
		ofs << env.instPath.substr(9, env.instPath.length() - 14) << " " << trueSets.size() << " " << delSets.size() << "\n";

		cout << "after dataReductionRule1:" << old_atLeastTrueSets.size() << endl;
		
		cout << trueSets.size() << endl;
		cout << delSets.size() << endl;

		getBelongTo(coverSets, belongTo,trueSets,delSets,residualNodesMap);

		for (auto iter = old_atLeastTrueSets.begin(); iter != old_atLeastTrueSets.end(); ++iter) {
			vector<ID> temp;
			for (auto i = 0; i < (*iter).size(); ++i) {
				if (residualNodesMap.isConsecutiveIdExist((*iter)[i])) {
					temp.push_back(residualNodesMap.toConsecutiveId((*iter)[i]));
				}
				else {
					cout << "error";
					exit(4);
				}
			}
			new_atLeastTrueSets.insert(temp);
		}

		string file_name = env.instPath.substr(9, env.instPath.length() - 14) + "_" + to_string(Radius) + ".cnf";
		string CNF_output;
		string Reuslt_path;

		Encoding_Mode mode = paralle_counter_l;

		if (mode == sequential_counter_1) {
			CNF_output = "CNF/sequential_counter_less/" + file_name;
			Reuslt_path = "Result\\sequential_counter_less\\glu_mix\\";
		}
		else if (mode == paralle_counter_l) {
			CNF_output = "CNF/paralle_counter_less/" + file_name;
			Reuslt_path = "Result\\paralle_counter_less\\glu_mix\\";
		}
		else if (mode == cardinality_network) {
			CNF_output = "CNF/cardinality_network/" + file_name;
			Reuslt_path = "Result\\cardinality_network\\";
		}
		int new_nodeNum = nodeNum - trueSets.size() - delSets.size();
		int new_centerNum = centerNum - trueSets.size();
		//sequential_counter_less(new_nodeNum, new_centerNum, file_name, belongTo, new_atLeastTrueSets);/*使用sequential_counter无冗余约束进行编码*/
		paralle_counter_less(new_nodeNum, new_centerNum, file_name, belongTo, new_atLeastTrueSets);/*使用sequential_counter无冗余约束进行编码*/
		//cardinalityNetwork(nodeNum, centerNum, file_name, belongTo);
		int res = 0;
		res = System::exec("glu_mix.exe " + CNF_output + " Result/" + file_name);/*对编码进行约束满足求解*/
		//solveWithBoop(file_name, mode, CNF_output, nodeNum);
		if (res == 10) {
			System::exec("copy Result\\" + file_name + " " + Reuslt_path);

			//Reuslt_path = "";
			ifstream in(Reuslt_path + file_name);

			delim = " ";
			in.getline(readLine, 100);
			in.getline(readLine, 100000);
			strcpy(tempLine, readLine);
			p = strtok(readLine, delim);
			int cnt = 0;
			for (int i = 0; i < new_nodeNum; ++i) {
				int temp = atoi(p);
				
				if (i == temp-1) {
					if (cnt == centerNum) {
						cout << "error" << endl;
						exit(-2);
					}
					centers[cnt++] = residualNodesMap.toArbitraryId(temp - 1);

				}
				p = strtok(NULL, delim);
				
			}
			for (auto iter = trueSets.begin(); iter != trueSets.end(); ++iter) {
				centers[cnt++] = *iter;
			}
			if (cnt < centerNum) {/* 使用更少的中心数可以满足，需要随便添加一些中心 */
				/*p = strtok(tempLine, delim);
				for (int i = 0; i < new_nodeNum && cnt < centerNum; ++i) {
					int temp = atoi(p);
					if (i + 1 == -temp) {
						centers[cnt++] = residualNodesMap.toArbitraryId(i);
					}
					p = strtok(NULL, delim);
				}*/
				for (auto iter = delSets.begin(); iter != delSets.end() && cnt <centerNum; ++iter) {
					centers[cnt++] = *iter;
				}
			}

			for (ID n = 0; n < nodeNum; ++n) {
				for (auto c = centers.begin(); c != centers.end(); ++c) {
					if (aux.adjMat.at(n, *c) < aux.coverRadii[n]) { aux.coverRadii[n] = aux.adjMat.at(n, *c); }
				}
				if (sln.coverRadius < aux.coverRadii[n]) { sln.coverRadius = aux.coverRadii[n]; }
			}
			Log(LogSwitch::Szx::Framework) << "worker " << workerId << " ends." << endl;
		}
		else {
			Sampling sampler(rand, centerNum);
			for (ID n = 0; !timer.isTimeOut() && (n < nodeNum); ++n) {
				ID center = sampler.replaceIndex();
				if (center >= 0) { centers[center] = n; }
			}
			//break;
		}

		//}


		return true;
	}



	void Solver::sequential_counter_less(int nodeNum, int centerNum, const string &file_name, vector<vector<ID>> &belongTo, set<vector<ID>> &atLeasttrueSets)
	{
		ofstream ofs;
		string CNF_output = "CNF/sequential_counter_less/" + file_name;
		ofs.open(CNF_output, ios::out);
		if (!ofs.is_open()) {
			cout << "[Fatal] Invalid answer path!" << endl;
		}
		else {
			int variable_num = nodeNum + (nodeNum - 1)*centerNum;
			int clauses_num = nodeNum + 2 * nodeNum*centerNum + nodeNum - 3 * centerNum - 1;
			for (int i = 0; i < nodeNum; ++i) {
				if (belongTo[i].size() ==  0) {
					clauses_num--;
				}
			}
			ofs << "p cnf " << variable_num << " " << clauses_num << endl;
			/*1~n 为x1到xn，表示是否选中该集合 */
			for (int i = 0; i < nodeNum; ++i) {
				if (belongTo[i].size() > 0) {
					for (int j = 0; j < belongTo[i].size(); ++j) {
						ofs << belongTo[i][j] + 1 << " ";
					}
					ofs << "0" << endl;
				}
				
			}
			/* n+1到n+p 为s1,1 到s1,p */
			ofs << "-1 " << nodeNum + 1 << " 0" << endl;
			for (int i = 1; i < centerNum; ++i) {
				ofs << "-" << nodeNum + 1 + i << " 0" << endl;
			}
			/* n+p+1到n+2p 为 s2,1 到 s2,p ，依次类推*/
			for (int i = 2; i < nodeNum; ++i) {
				int s_i_1 = nodeNum + centerNum * (i - 1) + 1;
				int s_i1_1 = nodeNum + centerNum * (i - 2) + 1;
				int s_i1_p = nodeNum + centerNum * (i - 2) + centerNum;
				ofs << "-" << i << " " << s_i_1 << " 0" << endl;
				ofs << "-" << s_i1_1 << " " << s_i_1 << " 0" << endl;
				for (int j = 2; j <= centerNum; ++j) {
					int s_i_j = nodeNum + centerNum * (i - 1) + j;
					int s_i1_j1 = nodeNum + centerNum * (i - 2) + j - 1;
					int s_i1_j = nodeNum + centerNum * (i - 2) + j;
					ofs << "-" << i << " -" << s_i1_j1 << " " << s_i_j << " 0" << endl;
					ofs << "-" << s_i1_j << " " << s_i_j << " 0" << endl;
				}
				ofs << "-" << i << " -" << s_i1_p << " 0" << endl;
			}
			int s_n1_p = nodeNum + centerNum * (nodeNum - 2) + centerNum;
			ofs << "-" << nodeNum << " -" << s_n1_p << " 0" << endl;
			for (auto i = 0; i < atLeasttrueSets.size(); ++i) {
				auto iter = atLeasttrueSets.begin();
				for (auto j = 0; j < (*iter).size(); ++j) {
					ofs << (*iter)[j] +1 << " ";
				}
				ofs << "0" << endl;
			}

		}

	}
	void Solver::full_adder(ofstream &ofs, int a, int b, int c, int s_out, int c_out) {
		ofs << a << " " << b << " -" << c << " " << s_out << " 0" << endl;
		ofs << a << " -" << b << " " << c << " " << s_out << " 0" << endl;
		ofs << "-" << a << " " << b << " " << c << " " << s_out << " 0" << endl;
		ofs << "-" << a << " -" << b << " -" << c << " " << s_out << " 0" << endl;
		ofs << "-" << a << " -" << b << " " << c_out << " 0" << endl;
		ofs << "-" << a << " -" << c << " " << c_out << " 0" << endl;
		ofs << "-" << b << " -" << c << " " << c_out << " 0" << endl;
		paralle_clause_num += 7;
	}
	void Solver::half_adder(ofstream &ofs, int a, int b, int s_out, int c_out) {
		ofs << a << " -" << b << " " << s_out << " 0" << endl;
		ofs << "-" << a << " " << b << " " << s_out << " 0" << endl;
		ofs << "-" << a << " -" << b << " " << c_out << " 0" << endl;
		paralle_clause_num += 3;
	}
	vector<int> Solver::recursive_counter(ofstream &ofs, int n, int start, int end) {//start和end为变量的范围
		vector<int> my_res;
		if (n == 0)return my_res;
		int m = log(n) / log(2);
		int temp = pow(2, m);
		vector<int> c_in;
		if (n == 1) {
			my_res.push_back(start);
		}
		else if (n == 2) {
			int s_out = paralle_var_num++;
			int c_out = paralle_var_num++;
			half_adder(ofs, start, end, s_out, c_out);
			my_res.push_back(s_out);
			my_res.push_back(c_out);
		}
		else if (n == 3) {
			int s_out = paralle_var_num++;
			int c_out = paralle_var_num++;
			int a = start, b = start + 1, c = end;
			full_adder(ofs, a, b, c, s_out, c_out);
			my_res.push_back(s_out);
			my_res.push_back(c_out);
		}
		else {
			vector<int> res1, res2;
			res1 = recursive_counter(ofs, temp - 1, start, start + temp - 2);
			res2 = recursive_counter(ofs, n - temp, start + temp - 1, end - 1);
			for (auto i = 0; i < res1.size(); ++i) {
				int s_out = paralle_var_num++;
				int c_out = paralle_var_num++;
				if (i < res2.size()) {
					if (i == 0) {
						full_adder(ofs, res1[i], res2[i], end, s_out, c_out);
					}
					else {
						full_adder(ofs, res1[i], res2[i], c_in[i - 1], s_out, c_out);
					}
					my_res.push_back(s_out);
					c_in.push_back(c_out);
				}
				else {
					if (i == 0) {
						half_adder(ofs, res1[i], end, s_out, c_out);
					}
					else {
						half_adder(ofs, res1[i], c_in[i - 1], s_out, c_out);
					}
					my_res.push_back(s_out);
					c_in.push_back(c_out);
				}
				if (i == res1.size() - 1) {
					my_res.push_back(c_out);
				}
			}
		}
		return my_res;
	}
	void Solver::paralle_counter_less(int nodeNum, int centerNum, const std::string & file_name, std::vector<std::vector<ID>>& belongTo, set<vector<ID>> &atLeasttrueSets)
	{
		ofstream ofs;
		string CNF_output = "CNF//paralle_counter_less//" + file_name;
		ofs.open(CNF_output, ios::out);
		if (!ofs.is_open()) {
			cout << "[Fatal] Invalid answer path!" << endl;
		}
		else {
			int n = nodeNum;
			paralle_var_num = n + 1;
			paralle_clause_num = 0;
			vector<int> res, p_binary;
			int temp = centerNum;
			while (temp != 0)
			{
				p_binary.push_back(temp % 2);
				temp = temp / 2;
			}
			vector<vector<int>> compares;
			res = recursive_counter(ofs, n, 1, n);
			vector<int> compare;

			for (auto i = 0; i < res.size(); ++i) {
				if (i < p_binary.size()) {
					if (p_binary[i] == 0) {
						ofs << "-" << res[i] << " 0" << endl;
					}
					else {
						ofs << res[i] << " 0" << endl;
					}
				}
				else {
					ofs << "-" << res[i] << " 0" << endl;
				}
				paralle_clause_num++;
			}
			for (int i = 0; i < nodeNum; ++i) {
				if (belongTo[i].size() > 0) {
					for (int j = 0; j < belongTo[i].size(); ++j) {
						ofs << belongTo[i][j] + 1 << " ";
					}
					ofs << "0" << endl;
					paralle_clause_num++;
				}
				
			}
			for (auto i = 0; i < atLeasttrueSets.size(); ++i) {
				auto iter = atLeasttrueSets.begin();
				for (auto j = 0; j < (*iter).size(); ++j) {
					ofs << (*iter)[j] + 1 << " ";
				}
				ofs << "0" << endl;
				paralle_clause_num++;

			}
			std::ifstream t(CNF_output);
			std::string str((std::istreambuf_iterator<char>(t)),
				std::istreambuf_iterator<char>());
			ofstream _ofs;
			_ofs.open(CNF_output, ios::out);
			_ofs << "p cnf " << paralle_var_num - 1 << " " << paralle_clause_num << endl;
			_ofs << str;

		}
	}

	vector<int> Solver::HMerge(ofstream &ofs, vector<int>&a, vector<int> &b) {
		vector<int> res;
		if (a.size() == 1 && b.size() == 1) {
			int c1 = ++cardinality_var_num;
			int c2 = ++cardinality_var_num;
			ofs << "-" << a[0] << " -" << b[0] << " " << c2 << " 0" << endl;
			ofs << "-" << a[0] << " " << c1 << " 0" << endl;
			ofs << "-" << b[0] << " " << c1 << " 0" << endl;
			cardinality_clause_num += 3;

			res.push_back(c1);
			res.push_back(c2);

		}
		else {
			vector<int> odd_a, odd_b, even_a, even_b;
			vector<int> d, e;
			for (auto i = 0; i < a.size(); ++i) {
				if (i % 2 == 0) {
					odd_a.push_back(a[i]);
				}
				else {
					even_a.push_back(a[i]);
				}
			}
			for (auto i = 0; i < b.size(); ++i) {
				if (i % 2 == 0) {
					odd_b.push_back(b[i]);
				}
				else {
					even_b.push_back(b[i]);
				}
			}
			d = HMerge(ofs, odd_a, odd_b);
			e = HMerge(ofs, even_a, even_b);
			res.push_back(d[0]);
			for (auto i = 2; i < 2 * a.size(); ++i) {
				res.push_back(++cardinality_var_num);
			}
			res.push_back(e[e.size() - 1]);

			for (auto i = 1; i <= a.size() - 1; ++i) {
				int index = i - 1;
				ofs << "-" << d[index + 1] << " -" << e[index] << " " << res[2 * index + 2] << " 0" << endl;
				ofs << "-" << d[index + 1] << " " << res[2 * index + 1] << " 0" << endl;
				ofs << "-" << e[index] << " " << res[2 * index + 1] << " 0" << endl;
				cardinality_clause_num += 3;

			}

		}
		return res;
	}

	vector<int> Solver::HSort(ofstream &ofs, int n, int start, int end) {
		if (n == 2) {
			vector<int> a, b;
			a.push_back(start);
			b.push_back(end);
			return HMerge(ofs, a, b);
		}
		int current_n = n / 2;
		vector<int>res1, res2;
		res1 = HSort(ofs, current_n, start, start + current_n - 1);
		res2 = HSort(ofs, current_n, start + current_n, end);
		return HMerge(ofs, res1, res2);
	}

	vector<int> Solver::SMerge(ofstream &ofs, vector<int> &a, vector<int> &b) {
		vector<int> res;
		if (a.size() == 1 && b.size() == 1) {
			int c1 = ++cardinality_var_num;
			int c2 = ++cardinality_var_num;
			ofs << "-" << a[0] << " -" << b[0] << " " << c2 << " 0" << endl;
			ofs << "-" << a[0] << " " << c1 << " 0" << endl;
			ofs << "-" << b[0] << " " << c1 << " 0" << endl;
			cardinality_clause_num += 3;
			res.push_back(c1);
			res.push_back(c2);
		}
		else {
			vector<int> odd_a, odd_b, even_a, even_b;
			vector<int> d, e;
			for (auto i = 0; i < a.size(); ++i) {
				if (i % 2 == 0) {
					odd_a.push_back(a[i]);
				}
				else {
					even_a.push_back(a[i]);
				}
			}
			for (auto i = 0; i < b.size(); ++i) {
				if (i % 2 == 0) {
					odd_b.push_back(b[i]);
				}
				else {
					even_b.push_back(b[i]);
				}
			}
			d = SMerge(ofs, odd_a, odd_b);
			e = SMerge(ofs, even_a, even_b);
			res.push_back(d[0]);
			for (auto i = 1; i <= a.size(); ++i) {
				res.push_back(++cardinality_var_num);
			}

			for (auto i = 1; i <= a.size() / 2; ++i) {
				int index = i - 1;
				ofs << "-" << d[index + 1] << " -" << e[index] << " " << res[2 * index + 2] << " 0" << endl;
				ofs << "-" << d[index + 1] << " " << res[2 * index + 1] << " 0" << endl;
				ofs << "-" << e[index] << " " << res[2 * index + 1] << " 0" << endl;
				cardinality_clause_num += 3;
			}
		}
		return res;
	}
	void Solver::cardinalityNetwork(int nodeNum, int centerNum, const std::string &file_name, std::vector<std::vector<ID>>& belongTo) {
		ofstream ofs;
		string CNF_output = "CNF//cardinality_network//" + file_name;
		ofs.open(CNF_output, ios::out);
		if (!ofs.is_open()) {
			cout << "[Fatal] Invalid answer path!" << endl;
		}
		else {
			vector<int> res, p_binary;
			int temp = centerNum;
			while (temp != 0)
			{
				p_binary.push_back(temp % 2);
				temp = temp / 2;
			}
			int k;
			for (auto r = 1; r < 50; ++r) {
				k = pow(2, r);
				if (k > centerNum) {
					break;
				}
			}
			int m = ceil(nodeNum*1.0 / k);
			int nPlusM = m * k;
			cardinality_clause_num = 0;
			cardinality_var_num = nPlusM;

			for (auto i = nodeNum + 1; i <= nPlusM; ++i) {
				ofs << "-" << i << " 0" << endl;
				cardinality_clause_num++;

			}
			vector<int> d1, d2;
			d1 = HSort(ofs, k, 1 + (m - 1)*k, m*k);
			d2 = HSort(ofs, k, 1 + (m - 2)*k, (m - 1)*k);

			d1 = SMerge(ofs, d1, d2);
			d1.pop_back();
			for (auto i = m - 3; i >= 0; --i) {
				int start = 1 + i * k, end = k + i * k;
				d2 = HSort(ofs, k, start, end);
				d1 = SMerge(ofs, d1, d2);
				d1.pop_back();
			}
			ofs << "-" << d1[centerNum] << " 0" << endl;
			cardinality_clause_num++;
			for (int i = 0; i < nodeNum; ++i) {
				for (int j = 0; j < belongTo[i].size(); ++j) {
					ofs << belongTo[i][j] + 1 << " ";
				}
				ofs << "0" << endl;
				cardinality_clause_num++;
			}
			std::ifstream t(CNF_output);
			std::string str((std::istreambuf_iterator<char>(t)),
				std::istreambuf_iterator<char>());
			ofstream _ofs;
			_ofs.open(CNF_output, ios::out);
			_ofs << "p cnf " << cardinality_var_num << " " << cardinality_clause_num << endl;
			_ofs << str;
		}
	}

	void Solver::convertCNFtoDNF(const std::string &cnf_file, const std::string &dnf_file, int nodeNum) {
		std::ifstream t(cnf_file);
		ofstream ofs;
		ofs.open(dnf_file, ios::out);
		string str;
		int temp;
		getline(t, str);
		str = str.substr(6, str.length());
		ofs << str << endl;
		stringstream ss(str);
		int var_num;
		ss >> var_num;
		ofs << "-2" << endl;
		while (getline(t, str)) {
			int num = 0;
			for (auto i = 0; i < str.length(); ++i) {
				if (str[i] == ' ')
					num++;
			}
			str = str.substr(0, str.length() - 2);
			stringstream ss(str);
			ofs << num << " ";
			while (num--) {
				ss >> temp;
				int i = -temp;
				ofs << i;
				if (num != 0)
					ofs << " ";
			}
			ofs << endl;
		}
		for (auto i = 0; i < var_num; ++i) {
			if (i < nodeNum)
				ofs << "1";
			else
				ofs << "0";
			if (i != var_num - 1)
				ofs << " ";
		}
	}
	void Solver::solveWithBoop(const std::string &file_name, Encoding_Mode mode, const std::string &cnf_file, int nodeNum) {
		string dnf_output;
		string Reuslt_path;
		if (mode == sequential_counter_1) {
			dnf_output = "DNF/sequential_counter_less/" + file_name;
			Reuslt_path = "Result\\DNF\\sequential_counter_less\\";
		}
		else if (mode == paralle_counter_l) {
			dnf_output = "DNF/paralle_counter_less/" + file_name;
			Reuslt_path = "Result\\DNF\\paralle_counter_less\\";
		}
		else if (mode == cardinality_network) {
			dnf_output = "DNF/cardinality_network/" + file_name;
			Reuslt_path = "Result\\DNF\\cardinality_network\\";
		}
		convertCNFtoDNF(cnf_file, dnf_output, nodeNum);
	}
#pragma endregion Solver

}
