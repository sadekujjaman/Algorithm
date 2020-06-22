import java.io.*;
import java.math.*;
import java.util.*;


/**
 *
 * @author Saju
 *
 */

public class Template {

	private static int dx[] = { 1, 0, -1, 0 };
	private static int dy[] = { 0, -1, 0, 1 };

	private static final long INF = Long.MAX_VALUE;
	private static final int INT_INF = Integer.MAX_VALUE;
	private static final long NEG_INF = Long.MIN_VALUE;
	private static final int NEG_INT_INF = Integer.MIN_VALUE;
	private static final double EPSILON = 1e-10;

	private static final int MAX = 1007;
	private static final long MOD = 1000000007;

	private static final int MAXN = 100007;
	private static final int MAXA = 10000009;
	private static final int MAXLOG = 22;

	public static void main(String[] args) {

		InputReader in = new InputReader(System.in);
		PrintWriter out = new PrintWriter(System.out);

/*



*/

		/*
		 
		int n = in.nextInt();
			int arr[] = new int[n];
			
			for(int i = 0; i < n; i++) {
				arr[i] = in.nextInt();
			}
			
			Arrays.sort(arr);
			long count[] = new long[MAX];
			
			for(int i = 0; i < n; i++) {
				if(arr[i] < MAX) {
					count[arr[i]]++;
				}
			}
			
			long pow[] = new long[MAX];
			for(int i = 1; i < MAX; i++) {
				if(count[i] == 0) {
					break;
				}
				else {
					pow[i] = bigMod(2, count[i], MOD) - 1;
				}
			}
			long val = bigMod(2, n, MOD);
			
			long val1 = val;
			long val2 = (long)n;
			long r = 1;
			for(int i = 1; i < MAX; i++) {
				if(pow[i] != 0) {
					r = (pow[i] * r) % MOD;
					val1 = (val1 + (r * bigMod(2, val2 - count[i], MOD))) % MOD;
					val2 -= count[i];
					
				}
				else {
					break;
				}
			}
			ans.append(val1 + "\n");
		 
		 */
		
		out.flush();
		out.close();
		System.exit(0);
	}

	
	private static class Graph {
		private int node;
		private List<Edge>[] adj;

		Graph(int node) {
			this.node = node;
			adj = new ArrayList[this.node];
			for (int i = 0; i < node; i++) {
				adj[i] = new ArrayList<Edge>();
			}
		}

		void addEdge(int u, int v, long cost) {
			Edge e = new Edge(u, v, cost);
			Edge e1 = new Edge(v, u, cost);
			adj[u].add(e);
			adj[v].add(e1);
		}

		long[] calculateShortestPath(int source) {

			long dist[] = new long[node];
			PriorityQueue<Node> queue = new PriorityQueue<Node>(new Comparator<Node>() {

				@Override
				public int compare(Node o1, Node o2) {
					int val = Long.compare(o1.distance, o2.distance);
					return val;

				}
			});

			Arrays.fill(dist, INT_INF);

			queue.add(new Node(source, 0));
			dist[source] = 0;

			while (!queue.isEmpty()) {
				Node nn = queue.remove();
				int u = nn.node;
				long udist = nn.distance;
				if (udist > dist[u]) {
					continue;
				}
				for (Edge e : adj[u]) {
					int v = e.v;
					long vdist = e.cost;
					long arrive = udist + vdist;
					if (arrive < dist[v]) {
						dist[v] = arrive;
						queue.add(new Node(v, dist[v]));
						// System.out.println(u + " " + v + " " + vdist + " " + dist[v]);
					}

				}

			}
			return dist;

		}
	}

	private static class Node {
		int operation;
		int node;
		long distance;

		Node(int operation, int node, long distance) {
			this.operation = operation;
			this.node = node;
			this.distance = distance;
		}

		Node(int node, long distance) {
			this.node = node;
			this.distance = distance;
		}
	}

	private static class Edge {
		int u;
		int v;
		long cost;

		Edge(int u, int v, long cost) {
			this.u = u;
			this.v = v;
			this.cost = cost;
		}

		int getNeighbourIndex(int node) {
			if (this.u == node) {
				return v;
			}
			return u;
		}
	}

	private static boolean isPalindrome(String str) {
		StringBuilder sb = new StringBuilder();
		sb.append(str);
		String str1 = sb.reverse().toString();
		return str.equals(str1);
	}

	private static String reverseString(String str) {
		StringBuilder sb = new StringBuilder();
		sb.append(str);
		return sb.reverse().toString();
	}

	private static double distance(Point p1, Point p2) {
		long divx = p2.x - p1.x;
		long divy = p2.y - p1.y;
		divx = divx * divx;
		divy = divy * divy;
		return Math.sqrt(divx + divy);
	}

	private static String getBinaryStr(int n, int j) {
		String str = Integer.toBinaryString(n);
		int k = str.length();
		for (int i = 1; i <= j - k; i++) {
			str = "0" + str;
		}

		return str;
	}

	// O(log(max(A,M)))
	static long modInverse(long a, long m) {
		extendedEuclid(a, m);
		return (x % m + m) % m;
	}

	static long x;
	static long y;
	static long gcdx;

	private static void extendedEuclid(long a, long b) {
		if (b == 0) {
			gcdx = a;
			x = 1;
			y = 0;
		} else {
			extendedEuclid(b, a % b);
			long temp = x;
			x = y;
			y = temp - ((a / b) * y);
		}
	}

	private static void generatePrime(int n) {
		// O(NloglogN)
		boolean arr[] = new boolean[n + 5];
		Arrays.fill(arr, true);
		for (int i = 2; i * i <= n; i++) {
			if (arr[i] == true) {
				for (int j = i * i; j <= n; j += i) {
					arr[j] = false;
				}
			}
		}
		int count = 0;
		int start = 0;
		for (int i = 2; i <= n; i++) {
			if (arr[i] == true) {
				// System.out.println(i + " ");
				count++;
			}
			if (count == (start * 100) + 1) {
				// System.out.println(i);
				start++;
			}
		}
		System.out.println();
		System.out.println(count);

	}

	private static Map<Long, Long> primeFactorization(long n, long m) {
		Map<Long, Long> map = new HashMap<>();

		for (long i = 2; i <= Math.sqrt(n); i++) {
			if (n % i == 0) {
				long count = 0;
				while (n % i == 0) {
					count++;
					n = n / i;
				}
				long val = count * m;
				map.put(i, val);
				// System.out.println("i: " + i + ", count: " + count);
			}

		}
		if (n != 1) {
			// System.out.println(n);
			map.put(n, m);
		}
		return map;
	}

private static class KruskalGraph{
		
		int vertices;
		int edges;
		int edgeCount = 0;
		Edge[] edgeArray;
		
		KruskalGraph(int n, int m){
			this.vertices = n;
			this.edges = m;
			edgeArray = new Edge[m];
			for(int i = 0; i < m; i++) {
				edgeArray[i] = new Edge();
			}
		}
		
		int find(Subset[] subsets, int i) {
			if(subsets[i].parent != i) {
				subsets[i].parent = find(subsets, subsets[i].parent);
			}
			return subsets[i].parent;
		}
		
		void union(Subset[] subsets, int x, int y) {
			int xRoot = find(subsets, x);
			int yRoot = find(subsets, y);
			
			int xRank = subsets[xRoot].rank;
			int yRank = subsets[yRoot].rank;
			
			if(xRank < yRank) {
				subsets[xRoot].parent = yRoot;
			}
			else if(yRank < xRank) {
				subsets[yRoot].parent = xRoot;
			}
			else {
				subsets[yRoot].parent = xRoot;
				subsets[xRoot].rank++;
			}
		}
		
		public double kruskalMST() {
			Edge[] result = new Edge[vertices];
			Subset[] subsets = new Subset[vertices];
			int e = 0;
			
			for(int i = 0; i < vertices; i++) {
				result[i] = new Edge();
				subsets[i] = new Subset(i, 0);
			}
			
			Arrays.sort(edgeArray);
			
			int i = 0;
			while(e < vertices - 1) {
				Edge nextEdge = new Edge();
				nextEdge = edgeArray[i++];
				int x = find(subsets, nextEdge.src);
				int y = find(subsets, nextEdge.dest);
				if(x != y) {
					result[e++] = nextEdge;
					union(subsets, x, y);
				}
			}
			
			double cost = 0.0;
			for(int j = 0; j < e; j++) {
				cost += result[j].weight;
			}
			return cost;
		}

		void addEdge(int src, int dest, double weight) {
			edgeArray[edgeCount++] = new Edge(src, dest, weight);
		}
		
		
		class Edge implements Comparable<Edge>{
			int src;
			int dest;
			double weight;
			
			Edge(){
				src = 0;
				dest = 0;
				weight = 0.0;
			}
			Edge(int src, int dest, double weight){
				this.src = src;
				this.dest = dest;
				this.weight = weight;
			}
			@Override
			public int compareTo(Edge edge) {
				return Double.compare(this.weight, edge.weight);
			}
		}
		
		class Subset{
			int parent;
			int rank;
			
			Subset(){
				
			}
			
			Subset(int parent, int rank){
				this.parent = parent;
				this.rank = rank;
			}
		}
	}
	
	private static class Pair<T> {
		T a;
		T b;

		Pair(T a, T b) {
			this.a = a;
			this.b = b;
		}
	}
	
	
	private static class DSU{
		int n;
		int[] parent;
		int[] rank;
		int ans = 0;
		DSU(int n) {
			this.n = n;
			this.parent = new int[n];
			this.rank = new int[n];
			makeSet();
		}
		private void makeSet() {
			for(int i = 0; i < n; i++) {
				parent[i] = i;
				rank[i] = 1;
			}
		}
		void union(int x, int y) {
			int parentX = parent(x);
			int parentY = parent(y);
			if(parentX != parentY) {
				if(rank[parentX] >= rank[parentY]) {
					rank[parentX] += rank[parentY];
					rank[parentY] = 0;
					parent[parentY] = parentX;
				}
				else {
					rank[parentY] += rank[parentX];
					rank[parentX] = 0;
					parent[parentX] = parentY;
				}
				ans -= 1;
			}
		}
		private int parent(int x) {
			if(parent[x] != x) {
				parent[x] = parent(parent[x]);
			}
			
			return parent[x];
		}
		
	}
	
	private static class HLD {
		final int MAX = 10001;
		final int MAXLOG = 15;

		int vertex;
		List<Edge> adj[] = new ArrayList[MAX];
		int[][] pp = new int[MAXLOG][MAX];
		int[] level = new int[MAX];
		long[] tree = new long[MAX << 2];
		int[] twoPower = new int[MAXLOG];

		int chainNo;
		int indexNo;
		int[] chainHead = new int[MAX];
		int[] chainSize = new int[MAX];
		int[] nodeInWhichChain = new int[MAX];
		int[] nodeIndexInBaseArray = new int[MAX];
		int[] baseArray = new int[MAX];
		int[] edgeIndex = new int[MAX];
		int[] subtreeSize = new int[MAX];

		public HLD() {
			for (int i = 0; i < MAX; i++) {
				adj[i] = new ArrayList<>();
			}
			twoPower[0] = 1;
			for (int i = 1; i < MAXLOG; i++) {
				twoPower[i] = twoPower[i - 1] << 1;
			}
		}

		public void clear() {
			for (int i = 1; i <= vertex; i++) {
				adj[i].clear();
			}
			for (int i = 1; i <= vertex; i++) {
				chainHead[i] = -1;
				chainSize[i] = -1;
			}

			for (int i = 0; i < MAXLOG; i++) {
				for (int j = 1; j <= vertex; j++) {
					pp[i][j] = -1;
				}
			}
		}

		public void addEdge(int u, int v, int w, int idx) {
			adj[u].add(new Edge(v, w, idx));
			adj[v].add(new Edge(u, w, idx));
		}

		public void buildHLD() {

			indexNo = 0;
			chainNo = 1;
			dfs(1, -1, 0);
			sparseTable();
			hld(1, -1, 0);
			build(1, 1, vertex);
//			
//			for(int i = 1; i <= 20; i++) {
//				System.out.print(baseArray[i] + " ");
//			}
//			System.out.println();
//			for(int i = 1; i <= 20; i++) {
//				System.out.print(tree[i] + " ");
//			}
//			System.out.println();
		}

		public void update(int idx, int val) {
			update(1, 1, vertex, edgeIndex[idx], val);
		}

		public long query(int u, int v) {
			int lca = lca(u, v);
			long val1 = hldQuery(lca, u);
			long val2 = hldQuery(lca, v);
			return val1 + val2;
		}
		
		public int getKthNode(int u, int v, int k){
			int lca = lca(u, v);
			
			if(level[u] - level[lca] <= k - 1){
				int temp = u;
				u = v;
				v = temp;
				k = 1 + level[u] + level[v] - (2 * level[lca]) - k;
			}else{
				k--;
			}
			
			for(int i = 0; (1<<i) <= k; i++) {
		        if((k & (1 << i)) >= 1) {
		            u = pp[i][u];
		        }
		    }
			
			return u;
		}
		private long hldQuery(int u, int v) {
			if (u == v) {
				return 0;
			}

			long ans = 0;
			while (true) {
				if (nodeInWhichChain[u] == nodeInWhichChain[v]) {
					int start = nodeIndexInBaseArray[u];
					int end = nodeIndexInBaseArray[v];
					if (start < end) {
//						ans = max(ans, query(1, 1, vertex, start + 1, end));
						ans += query(1, 1, vertex, start + 1, end);
					}
					return ans;
				}
				int head = chainHead[nodeInWhichChain[v]];
				int start = nodeIndexInBaseArray[head];
				int end = nodeIndexInBaseArray[v];
//				ans = max(ans, query(1, 1, vertex, start, end));
				ans += query(1, 1, vertex, start, end);
				v = pp[0][head];
			}
		}

		private void dfs(int u, int parent, int d) {
			pp[0][u] = parent;
			subtreeSize[u] = 1;
			level[u] = d;
			for (Edge e : adj[u]) {
				if (e.v != parent) {
					dfs(e.v, u, d + 1);
					subtreeSize[u] += subtreeSize[e.v];
				}
			}
		}

		private void sparseTable() {
			for (int i = 1; i < MAXLOG; i++) {
				for (int j = 1; j <= vertex; j++) {
					if (pp[i - 1][j] != -1) {
						pp[i][j] = pp[i - 1][pp[i - 1][j]];
					}
				}
			}
		}

		private void hld(int u, int parent, int cost) {

			indexNo++;
			nodeIndexInBaseArray[u] = indexNo;
			nodeInWhichChain[u] = chainNo;
			baseArray[indexNo] = cost;
			chainSize[chainNo]++;
			if (chainHead[chainNo] == -1) {
				chainHead[chainNo] = u;
			}

			int specialChild = -1;
			int specialCost = -1;
			int specialIdx = -1;
			int maxSubtreeSize = 0;

			for (Edge e : adj[u]) {
				int v = e.v;
				if (v != parent) {
					if (maxSubtreeSize < subtreeSize[v]) {
						maxSubtreeSize = subtreeSize[v];
						specialChild = v;
						specialCost = e.w;
						specialIdx = e.idx;
					}
				}
			}

			if (specialChild != -1) {
				edgeIndex[specialIdx] = indexNo + 1;
				hld(specialChild, u, specialCost);
			}

			for (Edge e : adj[u]) {
				int v = e.v;
				if (v != parent && v != specialChild) {
					chainNo++;
					edgeIndex[e.idx] = indexNo + 1;
					hld(v, u, e.w);
				}
			}
		}

		private void build(int node, int left, int right) {
//			System.out.println(node + " " + left + " " + right);
			if (left == right) {
				tree[node] = baseArray[left];
				return;
			}

			int mid = (left + right) >> 1;
			int leftNode = node << 1;
			int rightNode = leftNode | 1;
			build(leftNode, left, mid);
			build(rightNode, mid + 1, right);
			tree[node] = tree[leftNode] + tree[rightNode];//max(tree[leftNode], tree[rightNode]);
		}

		private void update(int node, int left, int right, int idx, int val) {
			if (left == right && idx == left) {
				tree[node] = val;
			} else {
				int mid = (left + right) >> 1;
				int leftNode = node << 1;
				int rightNode = leftNode | 1;
				if (idx <= mid) {
					update(leftNode, left, mid, idx, val);
				} else {
					update(rightNode, mid + 1, right, idx, val);
				}
//				tree[node] = max(tree[leftNode], tree[rightNode]);
				tree[node] = tree[leftNode] + tree[rightNode];
			}
		}

		private long query(int node, int left, int right, int start, int end) {

			if (left > right || left > end || right < start) {
				return 0;
			}

			if (left >= start && right <= end) {
				return tree[node];
			}

			int mid = (left + right) >> 1;
			int leftNode = node << 1;
			int rightNode = leftNode | 1;
			long val1 = query(leftNode, left, mid, start, end);
			long val2 = query(rightNode, mid + 1, right, start, end);

			return val1 + val2;
		}

		private int lca(int p, int q) {
			
			if(p == q) {
				return p;
			}
			
			if (level[p] < level[q]) {
				int temp = p;
				p = q;
				q = temp;
			}

			for (int i = MAXLOG - 1; i >= 0; i--) {
				if ((level[p] - twoPower[i]) >= level[q]) {
					p = pp[i][p];
				}
			}
			if (p == q) {
				return p;
			}

			for (int i = MAXLOG - 1; i >= 0; i--) {
				if (pp[i][p] != pp[i][q] && pp[i][p] != -1) {
					p = pp[i][p];
					q = pp[i][q];
				}
			}
			return pp[0][p];
		}

		private static class Edge {
			int v;
			int w;
			int idx;

			Edge() {

			}

			Edge(int v, int w, int idx) {
				this.v = v;
				this.w = w;
				this.idx = idx;
			}
		}
	}
	

	private static class SegmentTree {
		int n;
		private final int MAXN = 100007;
		private long[] tree = new long[MAXN << 2];
		private long[] lazy = new long[MAXN << 2];
		private long[] arr = new long[MAXN];

		/***
		 * 
		 * arr is 1 based index.
		 * 
		 * @param arr
		 * 
		 */
		SegmentTree(int n, long[] arr) {
			this.n = n;
			for (int i = 0; i < arr.length; i++) {
				this.arr[i] = arr[i];
			}
		}

		void build(int index, int left, int right) {
			if (left == right) {
				tree[index] = arr[left];
			} else {
				int mid = (left + right) / 2;

				build(index * 2, left, mid);
				build((index * 2) + 1, mid + 1, right);
				tree[index] = max(tree[(index * 2)], tree[(index * 2) + 1]);
			}
		}

		long query(int node, int left, int right, int start, int end) {

			if (left > end || right < start) {
				return NEG_INF;
			}

			if (left >= start && right <= end) {
				return tree[node];
			}
			int mid = (left + right) / 2;
			long val1 = query(2 * node, left, mid, start, end);
			long val2 = query(2 * node + 1, mid + 1, right, start, end);

			return max(val1, val2);
		}

		void update(int node, int left, int right, int idx, long val) {
			if (left == right) {
				tree[node] += val;
			} else {
				int mid = (left + right) / 2;
				if (idx <= mid) {
					update(2 * node, left, mid, idx, val);
				} else {
					update(2 * node + 1, mid + 1, right, idx, val);
				}
				tree[node] = max(tree[(2 * node) + 1], tree[(2 * node)]);
			}
		}

		void updateRange(int node, int start, int end, int l, int r, long val) {
			if (lazy[node] != 0) {
				// This node needs to be updated
				tree[node] = lazy[node]; // Update it
				if (start != end) {
					lazy[node * 2] = lazy[node]; // Mark child as lazy
					lazy[node * 2 + 1] = lazy[node]; // Mark child as lazy
				}
				lazy[node] = 0; // Reset it
			}
			if (start > end || start > r || end < l) // Current segment is not within range [l, r]
				return;
			if (start >= l && end <= r) {
				// Segment is fully within range
				tree[node] = val;
				if (start != end) {
					// Not leaf node
					lazy[node * 2] = val;
					lazy[node * 2 + 1] = val;
				}
				return;
			}
			int mid = (start + end) / 2;
			updateRange(node * 2, start, mid, l, r, val); // Updating left child
			updateRange(node * 2 + 1, mid + 1, end, l, r, val); // Updating right child
//			tree[node] = max(tree[node * 2], tree[node * 2 + 1]); // Updating root with max value
		}

		long queryRange(int node, int start, int end, int l, int r) {
			if (start > end || start > r || end < l)
				return 0; // Out of range
			if (lazy[node] != 0) {
				// This node needs to be updated
				tree[node] = lazy[node]; // Update it
				if (start != end) {
					lazy[node * 2] = lazy[node]; // Mark child as lazy
					lazy[node * 2 + 1] = lazy[node]; // Mark child as lazy
				}
				lazy[node] = 0; // Reset it
			}
			if (start >= l && end <= r) // Current segment is totally within range [l, r]
				return tree[node];
			int mid = (start + end) / 2;
			long p1 = queryRange(node * 2, start, mid, l, r); // Query left child
			long p2 = queryRange(node * 2 + 1, mid + 1, end, l, r); // Query right child
			return max(p1, p2);
		}

		void buildRange(int node, int low, int high) {
			if (low == high) {
				tree[node] = arr[low];
				return;
			}

			int mid = (low + high) / 2;
			int left = node << 1;
			int right = left | 1;
			buildRange(left, low, mid);
			buildRange(right, mid + 1, high);
			tree[node] = max(tree[left], tree[right]);
		}

		void printSegmentTree() {
			System.out.println(Arrays.toString(tree));
		}

	}

	private static class BIT {

		int[] BIT;// = new int[SIZE + 1];
		int n;

		// BIT 1 based indexing
		public BIT(int n) {
			this.n = n;
			BIT = new int[n + 1];
		}

		public int query(int index) {
			int sum = 0;
			for (; index > 0; index -= (index & (-index))) {
				sum += BIT[index];
			}
			return sum;
		}

		public void update(int index, int val) {
			for (; index <= n; index += (index & (-index))) {
				BIT[index] += val;
			}
		}
	}

	private static class KMP {

		private static char[] t;
		private static char[] s;

		public int kmp(char[] t, char[] s) {
			this.t = t;
			this.s = s;
			return this.kmp();
		}

		private int kmp() {

			List<Integer> prefixTable = getPrefixTable(s);

			int match = 0;
			int i = 0;
			int j = 0;

			int n = t.length;
			int m = s.length;
			while (i < n) {
				if (t[i] == s[j]) {
					if (j == m - 1) {
						match++;
						j = prefixTable.get(j - 1);
						continue;
					}
					i++;
					j++;
				} else if (j > 0) {
					j = prefixTable.get(j - 1);
				} else {
					i++;
				}
			}

			return match;
		}

		/***
		 * 1. We compute the prefix values π[i] in a loop by iterating <br/>
		 * from i=1 to i=n−1 (π[0] just gets assigned with 0). <br/>
		 * <br/>
		 * 2. To calculate the current value π[i] we set the variable j <br/>
		 * denoting the length of the best suffix for i−1. Initially j=π[i−1]. <br/>
		 * 3. Test if the suffix of length j+1 is also a prefix by <br/>
		 * <br/>
		 * comparing s[j] and s[i]. If they are equal then we assign π[i]=j+1, <br/>
		 * otherwise we reduce j to π[j−1] and repeat this step. <br/>
		 * <br/>
		 * 4. If we have reached the length j=0 and still don't have a match, <br/>
		 * then we assign π[i]=0 and go to the next index i+1. <br/>
		 * <br/>
		 * 
		 * @param pattern(String)
		 ***/
		private List<Integer> getPrefixTable(char[] pattern) {

			List<Integer> prefixTable = new ArrayList<Integer>();
			int n = pattern.length;
			for (int i = 0; i < n; i++) {
				prefixTable.add(0);
			}

			for (int i = 1; i < n; i++) {
				for (int j = prefixTable.get(i - 1); j >= 0;) {
					if (pattern[j] == pattern[i]) {
						prefixTable.set(i, j + 1);
						break;
					} else if (j > 0) {
						j = prefixTable.get(j - 1);
					} else {
						break;
					}
				}
			}

			return prefixTable;
		}
	}
	
	private static class MO {

		private final int MAXA = 1000007;
		private int answer = 0;
		private final int blockSize;

		private int n;
		private int arr[];
		private int q;
		private Query queries[];
		private int frequency[];
		private int ans[];
		private int qC = 0;

		public MO(int n, int arr[], int q) {
			this.n = n;
			this.arr = new int[n];
			for (int i = 0; i < n; i++) {
				this.arr[i] = arr[i];
			}
			this.q = q;
			queries = new Query[q];
			int sz = (int) Math.sqrt(n);
			if (sz * sz != n) {
				sz++;
			}
			blockSize = sz;
			frequency = new int[MAXA];
			ans = new int[q];
		}

		public void addQuery(int index, int left, int right) {
			queries[qC++] = new Query(index, left, right);
		}

		private void runMOS() {
			Arrays.sort(queries, new Comparator<Query>() {

				@Override
				public int compare(Query q1, Query q2) {
					int q1Block = q1.left / blockSize;
					int q2Block = q2.left / blockSize;
					if (q1Block == q2Block) {
						return q1.right - q2.right;
					}
					return q1Block - q2Block;
				}
			});

			int currentLeft = queries[0].left;
			int currentRight = queries[0].left;

			frequency[arr[currentLeft]]++;
			answer = 1;

			for (int i = 0; i < q; i++) {
				Query query = queries[i];
				int left = query.left;
				int right = query.right;

				while (currentLeft < left) {
					remove(currentLeft);
					currentLeft++;
				}

				while (currentLeft > left) {
					add(currentLeft - 1);
					currentLeft--;
				}

				while (currentRight < right) {
					add(currentRight + 1);
					currentRight++;
				}
				while (currentRight > right) {
					remove(currentRight);
					currentRight--;
				}
//				System.out.println(currentLeft + " " + currentRight);
				ans[query.index] = answer;
			}
		}

		public String getAns() {
			runMOS();
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < q; i++) {
				sb.append(ans[i] + "\n");
			}
			return sb.toString();
		}

		private void add(int index) {
			frequency[arr[index]]++;
			if (frequency[arr[index]] == 1) {
				answer++;
			}
		}

		private void remove(int index) {
			frequency[arr[index]]--;
			if (frequency[arr[index]] == 0) {
				answer--;
			}
		}

		private class Query {
			int index;
			int left;
			int right;

			Query(int index, int left, int right) {
				this.index = index;
				this.left = left;
				this.right = right;
			}
		}
	}
	

	private static class Point {
		long x;
		long y;

		Point(long x, long y) {
			this.x = x;
			this.y = y;
		}

		@Override
		public boolean equals(Object obj) {
			Point ob = (Point) obj;
			if (this.x == ob.x && this.y == ob.y) {
				return true;
			}
			return false;

		}

		@Override
		public String toString() {
			return this.x + " " + this.y;
		}

		@Override
		public int hashCode() {
			return 0;
		}
	}

	private static long pow(int base, int pow) {
		long val = 1L;
		for (int i = 1; i <= pow; i++) {
			val *= base;
		}
		return val;
	}

	private static int log(int x, int base) {
		return (int) (Math.log(x) / Math.log(base));
	}

	private static int log(long x, int base) {
		return (int) (Math.log(x) / Math.log(base));
	}

	private static long max(long a, long b) {
		if (a >= b) {
			return a;
		}
		return b;
	}

	private static long abs(long a) {
		if (a < 0) {
			return -a;
		}
		return a;
	}

	private static int abs(int a) {
		if (a < 0) {
			return -a;
		}
		return a;
	}

	private static int max(int a, int b) {
		if (a >= b) {
			return a;
		}
		return b;
	}

	private static int min(int a, int b) {
		if (a <= b) {
			return a;
		}
		return b;
	}

	private static long min(long a, long b) {
		if (a <= b) {
			return a;
		}
		return b;
	}

	private static long gcd(long a, long b) {
		if (b == 0) {
			return a;
		}
		return gcd(b, a % b);
	}

	private static int gcd(int a, int b) {
		if (b == 0) {
			return a;
		}
		return gcd(b, a % b);
	}

	private static long bigMod(long n, long k, long m) {

		long ans = 1;
		while (k > 0) {
			if ((k & 1) == 1) {
				ans = (ans * n) % m;
			}
			n = (n * n) % m;
			k >>= 1;
		}
		return ans;
	}

	/*
	 * Returns an iterator pointing to the first element in the range [first, last]
	 * which does not compare less than val.
	 * 
	 */
	private static int lowerBoundNew(long[] arr, long num) {
		int start = 0;
		int end = arr.length - 1;
		int index = 0;
		int len = arr.length;
		int mid = 0;
		while (true) {
			if (start > end) {
				break;
			}
			mid = (start + end) / 2;
			if (arr[mid] > num) {
				end = mid - 1;
			} else if (arr[mid] < num) {
				start = mid + 1;
			} else {
				while (mid >= 0 && arr[mid] == num) {
					mid--;
				}
				return mid + 1;
			}
		}
		if (arr[mid] < num) {
			return mid + 1;
		}
		return mid;
	}

	/*
	 * upper_bound() is a standard library function in C++ defined in the header .
	 * It returns an iterator pointing to the first element in the range [first,
	 * last) that is greater than value, or last if no such element is found
	 * 
	 */
	private static int upperBoundNew(long arr[], long num) {

		int start = 0;
		int end = arr.length - 1;
		int index = 0;
		int len = arr.length;
		int mid = 0;
		while (true) {
			if (start > end) {
				break;
			}
			mid = (start + end) / 2;
			long val = arr[mid];
			if (val > num) {
				end = mid - 1;
			} else if (val < num) {
				start = mid + 1;
			} else {
				while (mid < len && arr[mid] == num) {
					mid++;
				}
				if (mid == len - 1 && arr[mid] == num) {
					return mid + 1;
				} else {
					return mid;
				}
			}
		}
		if (arr[mid] < num) {
			return mid + 1;
		}
		return mid;
	}

	private static class InputReader {
		public BufferedReader reader;
		public StringTokenizer tokenizer;

		public InputReader(InputStream stream) {
			reader = new BufferedReader(new InputStreamReader(stream));
			tokenizer = null;
		}

		public String next() {
			try {
				while (tokenizer == null || !tokenizer.hasMoreTokens()) {
					tokenizer = new StringTokenizer(reader.readLine());

				}
			} catch (IOException e) {
				return null;
			}
			return tokenizer.nextToken();
		}

		public String nextLine() {
			String line = null;
			try {
				tokenizer = null;
				line = reader.readLine();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
			return line;
		}

		public int nextInt() {
			return Integer.parseInt(next());
		}

		public double nextDouble() {
			return Double.parseDouble(next());
		}

		public long nextLong() {
			return Long.parseLong(next());
		}

		public boolean hasNext() {
			try {
				while (tokenizer == null || !tokenizer.hasMoreTokens()) {
					tokenizer = new StringTokenizer(reader.readLine());
				}
			} catch (Exception e) {
				return false;
			}
			return true;
		}
	}
}
