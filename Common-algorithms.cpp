#include <iostream>
#include <vector>
using namespace std;
//求最小公倍数 
int lcm(int a, int b) {
  return (a * b) / gcd(a, b);
}
// 求最大公约数
int gcd(int a, int b) {
  if (b == 0) {
    return a;
  } else {
    return gcd(b, a % b);
  }
  
  
//全排列
void permute(vector<int>& nums, int start, vector<vector<int>>& result) {
  if (start == nums.size() - 1) {
    result.push_back(nums);
    return;
  }

  for (int i = start; i < nums.size(); i++) {
    swap(nums[start], nums[i]);
    permute(nums, start + 1, result);
    swap(nums[start], nums[i]);
  }
}

 
// 前缀表
vector<int> prefix(const string &pattern) {
  int n = pattern.size();
  vector<int> pi(n);
  pi[0] = 0;
  int k = 0;
  for (int i = 1; i < n; i++) {
    while (k > 0 && pattern[i] != pattern[k]) {
      k = pi[k - 1];
    }
    if (pattern[i] == pattern[k]) {
      k++;
    }
    pi[i] = k;
  }
  return pi;
}

// KMP算法
vector<int> KMP(const string &text, const string &pattern) {
  vector<int> occurrences;
  int m = text.size();
  int n = pattern.size();
  vector<int> pi = prefix(pattern);
  int q = 0;  // 模式串中的位置
  for (int i = 0; i < m; i++) {
    while (q > 0 && pattern[q] != text[i]) {
      q = pi[q - 1];
    }
    if (pattern[q] == text[i]) {
      q++;
    }
    if (q == n) {
      occurrences.push_back(i - n + 1);
      q = pi[q - 1];
    }
  }
  return occurrences;
}



vector<int> divide(vector<int>& a, int b, int& remainder) {
    vector<int> res;
    remainder = 0;
    for (int i = a.size() - 1; i >= 0; i--) {
        remainder = remainder * 10 + a[i];
        res.push_back(remainder / b);
        remainder %= b;
    }
    reverse(res.begin(), res.end());
    while (res.size() > 1 && res.back() == 0) {
        res.pop_back();
    }
    return res;
}

int main() {
    string num;
    int divisor;
    cin >> num >> divisor;
    vector<int> a;
    for (int i = num.size() - 1; i >= 0; i--) {
        a.push_back(num[i] - '0');
    }
    int remainder;
    vector<int> res = divide(a, divisor, remainder);
    for (int i = res.size() - 1; i >= 0; i--) {
        cout << res[i];
    }
    cout << " " << remainder;
    return 0;
}
#include <iostream>
#include <vector>
using namespace std;

vector<int> add(vector<int>& a, vector<int>& b) {
    vector<int> res;
    int carry = 0;
    int n = max(a.size(), b.size());
    for (int i = 0; i < n; i++) {
        if (i < a.size()) carry += a[i];
        if (i < b.size()) carry += b[i];
        res.push_back(carry % 10);
        carry /= 10;
    }
    if (carry) res.push_back(carry);
    return res;
}

int main() {
    string num1, num2;
    cin >> num1 >> num2;
    vector<int> a, b;
    for (int i = num1.size() - 1; i >= 0; i--) {
        a.push_back(num1[i] - '0');
    }
    for (int i = num2.size() - 1; i >= 0; i--) {
        b.push_back(num2[i] - '0');
    }
    vector<int> res = add(a, b);
    for (int i = res.size() - 1; i >= 0; i--) {
        cout << res[i];
    }
    return 0;
}

#include <iostream>
#include <vector>
using namespace std;

vector<int> subtract(vector<int>& a, vector<int>& b) {
    vector<int> res;
    int n = max(a.size(), b.size());
    for (int i = 0, carry = 0; i < n; i++) {
        if (i < a.size()) carry += a[i];
        if (i < b.size()) carry -= b[i];
        if (carry < 0) {
            res.push_back(carry + 10);
            carry = -1;
        } else {
            res.push_back(carry);
            carry = 0;
        }
    }
    while (res.size() > 1 && res.back() == 0) {
        res.pop_back();
    }
    return res;
}

int main() {
    string num1, num2;
    cin >> num1 >> num2;
    vector<int> a, b;
    for (int i = num1.size() - 1; i >= 0; i--) {
        a.push_back(num1[i] - '0');
    }
    for (int i = num2.size() - 1; i >= 0; i--) {
        b.push_back(num2[i] - '0');
    }
    vector<int> res = subtract(a, b);
    for (int i = res.size() - 1; i >= 0; i--) {
        cout << res[i];
    }
    return 0;
}
#include <iostream>
#include <vector>
using namespace std;

vector<int> multiply(vector<int>& a, vector<int>& b) {
    vector<int> res(a.size() + b.size(), 0);
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b.size(); j++) {
            res[i + j] += a[i] * b[j];
            res[i + j + 1] += res[i + j] / 10;
            res[i + j] %= 10;
        }
    }
    while (res.size() > 1 && res.back() == 0) {
        res.pop_back();
    }
    return res;
}

int main() {
    string num1, num2;
    cin >> num1 >> num2;
    vector<int> a, b;
    for (int i = num1.size() - 1; i >= 0; i--) {
        a.push_back(num1[i] - '0');
    }
    for (int i = num2.size() - 1; i >= 0; i--) {
        b.push_back(num2[i] - '0');
    }
    vector<int> res = multiply(a, b);
    for (int i = res.size() - 1; i >= 0; i--) {
        cout << res[i];
    }
    return 0;
}

二分
vector<int> a;

int find(vector<int>& a, int as, int key) {
    if (a.empty() || as == 0) {
        return -1;
    }
    int l = 0, r = as - 1, mid = 0;
    while(l <= r) {
        mid = l + r >> 1;
        if (a[mid] < key) {
            l = mid + 1;
        } else if (a[mid] == key) {
            return mid;
        } else {
            r = mid - 1;
        }
    }
    return -1;
}

int find(vector<int>& a, int l, int r, int key) {
    if (l > r) {
        return -1;
    }
    int mid = l + r >> 1;
    if (a[mid] == key) {
        return mid;
    } else if (a[mid] < key) {
        return find(a, mid + 1, r, key);
    } else {
        return find(a, l, mid - 1, key);
    }
}

并查集
class DisjointSet{
private:
    vector<int> parent;
public:
    DisjointSet(int max_size) : parent(vector<int>(max_size)) {
        for (int i = 0; i < max_size; i++) { //initialize;
            parent[i] = i;
        }
    }
    int find(int x) {
        return x == parent[x] ? x : find(parent[x]);
    }
    
    void to_union(int x1, int x2) {
        parent[find(x1)] = find(x2);
    }
    
    bool is_same(int e1, int e2) {
        return find(e1) == find(e2);
    }
};


private:
    vector<int> parent;
    vector<int> rank;
public:
    DisjointSet(int max_size) : parent(vector<int>(max_size)),
                                rank(vector<int>(max_size, 0))
    {
        for (int i = 0; i < max_size; i++) { //initialize;
            parent[i] = i;
        }
        int find(int x) {
            return x == parent[x] ? x : find(parent[x]);
        }
        void union(int x1, int x2) {
            int f1 = find(x1), f2 = find(x2);
            if (rank[f1] < rank[f2]) {
                parent[f1] = f2;
            } else if (rank[f1] > rank[f2]) {
                parent[f2] = f1;
            } else {
                ++rank[f2];
            }
        }

        bool is_same(int e1, int e2) {
            return find(e1) == find(e2);
        }
    }
    int find(int x) {
        return x == parent[x] ? x : find(parent[x]);
    }

    void to_union(int x1, int x2) {
        parent[find(x1)] = find(x2);
    }

    bool is_same(int e1, int e2) {
        return find(e1) == find(e2);
    }
};

Floyd求最短路
int G[505][505]; //adjacency matrix
int n; // matrix

void init()
{
    memset(G,0x3f, sizeof(G)); //initialize
    for (int i = 1; i <= n; i++) {
        G[i][i] = 0; //path;
    }
}
void Floyd(int n) {
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (G[i][j] > G[i][k] + G[k][j]) {
                    G[i][j] = G[i][k] + G[k][j]
                }
            }
        }
    }
}

Floyd打印最短路径
int G[505][505] , path[505][505]; //adjacency matrix
int n; // matrix
const int inf = 0x7f7f7f7f;
void init()
{
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            G[i][j] = (i == j ? 0 : inf);
            path[i][j] = j;
        }
    }
}
void Floyd() {
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j<= n; j++) {
                if (G[i][j] > G[i][k] + G[k][j]) {
                    G[i][j] = G[i][k] + G[k][j];
                    path[i][j] = path[i][k];
                }
            }
        }
    }
}

void print(int u, int v) {
    if (u == v) {
        cout << v << endl; //A point
        return;
    }
    cout << u << " ";
    print(path[u][v], v);
}

Floyd求最小字典序的路径
int G[505][505], path[505][505];
int n;
void init() {
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            G[i][j] = (i == j ? 0 : inf);
            path[i][j] = j;
        }
    }
}

void Floyd() {
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (G[i][j] > G[i][k] + G[k][j]) {
                    G[i][j] = G[i][k] + G[k][j];
                    path[i][j] = path[i][k];
                } else if(G[i][j] == G[i][k] + G[k][j] && path[i][j] > path[i][k]) {
                    path[i][j] = path[i][k];
                }
            }
        }
    }
}

void print(int u, int v) {
    if (u == v) {
        cout<<v<<endl;
        return ;
    }
    cout<<u<<" ";
    print(path[u][v], v);
}

Floyd求最小环
void init() {
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            G[i][j] = d[i][j] = (i == j ? 0 : inf);
        }
    }
}

int Floyd() {
    int ans = inf;
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                ans = min(ans, G[i][k] + G[k][j] + d[i][j]);
            }
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
            }
        }
    }
    return ans;
}


Floyd求最短路的数量
int s, e, cnt;
int g[505][505], G[505][505];
void dfs(int u) {
	if (u == s) {
		cnt++;
		return ;
	}
	for (int i = 1; i <= n; i++) {
		if (i != u && G[u][i] < inf && g[s][u] == g[s][i] + G[u][i]) {
			dfs(i);
		}
	}
}
void solve() {
	dfs(e);
	cout<<cnt<<endl;
}


Floyd求传递闭包
int vis[505][505];
int n;
void Floyd() {
	for (int k = 1; k <= n; k++) {
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= n; j++) {
				if (vis[i][k] && vis[k][j]) {
					vis[i][j] == 1;
				}
			}
		}
	}
}


C++任意进制之间转换
#include <bits/stdc++.h>
using namespace std;
int n, m,  len, i,  t = 0, k;
char a[1000], b[1000];
int main()
{
	int count = 0;
	cin>>n>>m;
	cin>>a;
	len = strlen(a);
	k = len - 1;
	for (i = 0; i < len; i++) {
		if (a[i] >= '0' && a[i] <= '9') {
			t = a[i]  - '0'; // char to int
		} else {
			t = a[i] - 'A' + 10; //16
		} 
		count += t * pow(n, k) ;
		k--;
	}
	len = 0;
	while(count > 0) {
		t = count % m;
		count /= m;
		if (t >= 0 && t <= 9) {
			b[len] = '0' + t;
		} else {
			b[len] = 'A' + (t - 10) ;
		}
		len++;
	}
	for (int i = len - 1; i >= 0; i--) {
		cout<<b[i];
	}
	return 0;
}

Dijkstra算法
清除标记数组中所有点的标记
设d[s] = 0, 其他d[i] = INF, s是起点
循环n次 {
    在所有未标记节点中选出d值最小的节点k
    给节点k标记
    对于k标记
    对于从k出发的所有边（k, i), 更新d[i] = min(d[i], d[k] + w(k, i));
}
领接矩阵
const int INF = 0x3f3f3f3f;
const int N = 1e3 + 10;
int G[N][N]; //adjacency matrix
int d[N];
bool vis[N];
int n; //vertex

void Dijkstra(int u) {
    memset(d, 0x3f, sizeof(d));
    memset(vis, 0, sizeof(vis));
    d[u] = 0;
    for (int i = 1; i <= n; i++) {
        int k, m, INF;
        for (int j = 1; j <= n; j++) {
            if (!vis[j] && d[j] < m) {
                m = d[j];
                k = j;
            }
        }
        vis[k] = 1;
        for (int j = 1; j <= n; j++) {
            d[j] = min(d[j], d[k] + G[k][j]);
        }
    }
    for (int i = 1; i <= n; i++) {
        cout<<d[i]<<" ";
    }
    cout<<endl;
}

邻接表

typedef pair<int, int> P;
const int maxn = 1e5 + 10;
const int maxm = 2e5 + 10;

struct edge {
    int from, to, w;
    edge(int a, int b, int c) : from(a), to(b), w(c) {}
};

struct Dijkstra{
    int n, m; // 节点数和边数
    vector<edge> edges;
    vector<int> G[maxn];
    bool vis[maxm];
    int d[maxn];

    void init(int n) {
        this->n = n;
        for (int i = 0; i <= n; i++) {
            G[i].clear();
        }
    }

    void addEdge(int from, int to, int w) {
        edges.push_back(edge(from, to, w));
        m = edges.size();
        G[from].push_back(m - 1);
    }

    void dijkstra(int s) {
        priority_queue<P, vector<P>, greater<P> > q;
        memset(vis, 0, sizeof(vis));
        memset(d, 0x3f, sizeof(d));
        d[s] = 0;
        q.push(make_pair(0, s));
        while(!q.empty()) {
            P pr = q.top();
            q.pop();
            int u = pr.second;
            if (vis[u]) {
                continue;
            }
            vis[u] = 1;
            for (int i = 0; i < G[u].size(); i++) {
                edge &e = edges[G[u][i]];
                if (d[e.to] > d[u] + e.w) {
                    d[e.to] = d[u] + e.w;
                    q.push(make_pair(d[e.to], e.to));
                }
            }
        }
    }

    void print()
    {
        for (int i = 1; i <= n; i++) {
            cout<<d[i]<<" ";
        }
    }

链式前向星
typedef pair<int, int> P;
const int maxn = 1e5 + 10;
const int maxm = 2e5 + 10;

struct node {
    int to, next, w;
};
struct Dijkstra{
    int n;
    int tot;
    int head[maxn];
    node edge[maxm];
    int d[maxn];
    
    void init(int n) {
        this->n = n;
        tot = 0;
    }
    
    void addEdge(int u, int v, int w) {
        tot++;
        edge[tot].w = w;
        edge[tot].to = v;
        edge[tot].next = head[u];
        head[u]  = tot;
    }
    
    void dijkstra(int s) {
        priority_queue<P, vector<P>,greater<P> > q;
        memset(d, 0x3f, sizeof(d));
        d[s] = 0;
        q.push(make_pair(0, s));
        while(!q.empty()) {
            P pr = q.top();
            q.pop();
            int u = pr.second;
            if (pr.first != d[u]) {
                continue;
            }
            for (int i = head[u]; i; i = edge[i].next) {
                int v = edge[i].to;
                int w = edge[i].w;
                if (d[v] > d[u] + w) {
                    d[v] = d[u] + w;
                    q.push(make_pair(d[v], v));
                }
            }
        }
    }
    void print() {
        for (int i = 1; i <= n; i++) {
            cout<<d[i]<<" ";   
        }
    }
凸包算法
const int n=20;
const double MAXNUM=1e10;


class c_point
{
public:
    double x,y;
    c_point(){x=rand()%100;y=rand()%100;}//构造函数
};

c_point p0;//最低点
c_point points[n];//散点集
stack<c_point>convex_hull;//凸包

//显示栈
void check_stack(stack<c_point>points)
{
    cout<<"==========="<<endl;
    while(!points.empty()){
        cout<<"  ["<<points.top().x<<","<<points.top().y<<"]"<<endl;
        points.pop();
    }
    cout<<"==========="<<endl;
}

//寻找p0
void find_p0()
{
    p0=points[0];
    int ii=0;
    for(int i=1;i<n;i++){
        if(points[i].y<p0.y){
            p0=points[i];
            ii=i;
        }else if(points[i].y==p0.y){
            if(points[i].x<p0.x){
                p0=points[i];
                ii=i;
            }
        }
    }
}

//极角排序
bool cmp(c_point &p1,c_point &p2)
{
    //p0排首位
    if(p1.x==p0.x&&p1.y==p0.y)return true;
    if(p2.x==p0.x&&p2.y==p0.y)return false;

    //计算极角（等于0则赋予一个极大值）
    double angle1=p1.x==p0.x?MAXNUM:(p1.y-p0.y)/(p1.x-p0.x);
    double angle2=p2.x==p0.x?MAXNUM:(p2.y-p0.y)/(p2.x-p0.x);
    //小于0则赋予一个更大的值
    if(angle1<0)angle1+=2*MAXNUM;
    if(angle2<0)angle2+=2*MAXNUM;

    //极角排序
    if(angle1<angle2)return true;
    else if(angle1==angle2){
        if(p1.y>p2.y)return true;
        else return false;
    }
    else return false;
}

//叉积
double cross(c_point p1, c_point p2, c_point p3)
{
    return (p2.x-p1.x)*(p3.y-p1.y)-(p3.x-p1.x)*(p2.y-p1.y);
}

//搜索凸包
void find_convex_hull()
{
    //p0和p1是凸包中的点
    convex_hull.push(points[0]);
    convex_hull.push(points[1]);

    int i=2;
    //p1,p2为栈顶两个节点
    c_point p1=points[0];
    c_point p2=points[1];
    while(i<n){
        //如果points[i]和points[i-1]在同一个角度，则不再对points[i]进行计算
        if((points[i-1].y-p0.y)*(points[i].x-p0.x)==(points[i-1].x-p0.x)*(points[i].y-p0.y)){
            i++;
            continue;
        }

        //如果叉积大于0，将当前点压入栈
        if (cross(p1, p2, points[i])>=0){
            //假设现在栈中为a,b,c,d,cross(c,d,e)大于等于0
            convex_hull.push(points[i]);//a,b,c,d,e,p1=c,p2=d
            p1=p2;//p1=d
            p2=convex_hull.top();//p2=e
            i++;
        }

            //如果叉积小于0，对栈中节点进行处理
        else{
            while(1){
                //假设现在栈中为a,b,c,d,cross(c,d,e)小于0
                convex_hull.pop();//a,b,c
                convex_hull.pop();//a,b
                p2=p1;//p2=c;
                p1=convex_hull.top();//p1=b
                convex_hull.push(p2);//a,b,c
                //cross(b,c,e)
                if(cross(p1,p2,points[i])>=0){
                    convex_hull.push(points[i]);//a,b,c,e
                    p1=p2;//p1=c
                    p2=convex_hull.top();//p2=e
                    i++;
                    break;
                }
            }
        }
    }
}

int main()
{
    find_p0();//寻找p0
    sort(points,points+n,cmp);//按极角排序
    find_convex_hull();//搜索凸包
    check_stack(convex_hull);//显示结果

    system("pause");
}


Bellman-Ford

const int maxn=1e5+10; 	//点的最大值
const int maxm=2e5+10; 	//边的最大值
int d[maxn]; 			//起点到每个点的距离
int u[maxm],v[maxm],w[maxm]; //省略了结构体写法

void bellman_ford(int s){ //起点s
    memset(d,0x3f,sizeof(d)); //初始化到每个点都为INF
    d[s]=0;
    for(int i=1;i<=n-1;i++){
        for(int j=1;j<=m;j++){
            d[v[j]]=min(d[v[j]],d[u[j]]+w[j]);
        }
    }
}


GCD 
int gcd(int n, int m) {
	if (n % m == 0 ) 
		return m;
	return gcd(m, n % m);
}

快速幂
int qpow(int a, int n) {
    if (n == 0) {
        return 1;
    } else if (n % 2 == 1) {
        return qpow(a, n - 1) * a;
    } else {
        int temp = qpow(a, n / 2);
        return temp * temp;
    }
}

//next_permutation全排列
int n, i;
cin>>n;
int num[n];
for (i = 0; i < n; i++) num[i] = i + 1;
do{
	for(i = 0; i < n; i++)
		cout<<num[i];
	cout<<endl;
}while(next_permutation(num, num ++ n));
return 0;
字典树
class Trie {
private:
    bool isEnd;
    Trie* next[26];
public:
    Trie() {
        isEnd = false;
        memset(next, 0, sizeof(next));
    }
    
    void insert(string word) {
        Trie* node = this;
        for (char c : word) {
            if (node->next[c-'a'] == NULL) {
                node->next[c-'a'] = new Trie();
            }
            node = node->next[c-'a'];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        Trie* node = this;
        for (char c : word) {
            node = node->next[c - 'a'];
            if (node == NULL) {
                return false;
            }
        }
        return node->isEnd;
    }
    
    bool startsWith(string prefix) {
        Trie* node = this;
        for (char c : prefix) {
            node = node->next[c-'a'];
            if (node == NULL) {
                return false;
            }
        }
        return true;
    }
};

