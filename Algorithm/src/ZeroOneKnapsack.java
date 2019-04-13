

import java.util.Arrays;

public class ZeroOneKnapsack {

	static int weight[] = {2, 3, 4, 5};
	static int profit[] = {1, 2, 5, 6};
	static int capacity = 8;
	
	static int dp[][] = new int[weight.length + 1][capacity + 1];
	public static void main(String[] args) {
		
		for(int i = 0; i <= weight.length; i++){
			Arrays.fill(dp[i], -1);
		}
		int ans = call(0, 0);
		System.out.println(ans);
		System.out.print("   ");
		for(int i = 0; i <= capacity; i++){
			System.out.print(i + "   ");
		}
		System.out.println();
		for(int i = 0; i <= weight.length; i++){
			System.out.print(i + "  ");
			for(int j = 0; j <= capacity; j++){
				if(dp[i][j] != -1){
					System.out.print(dp[i][j] + "   ");
				}
				else{
					System.out.print( "x   ");
				}
				
				
			}
			System.out.println();
			
		}
		
		System.out.println();
		System.out.println();
		
		int table[][] = new int[weight.length + 1][capacity + 1];
		
		for(int i = 0; i <= weight.length; i++){
			for(int j = 0; j <= capacity; j++){
				if(i == 0 || j== 0){
					table[i][j] = 0;
					continue;
				}
				if(j < weight[i - 1]){
					table[i][j] = table[i - 1][j];
				}
				else{
					table[i][j] = Math.max(profit[i - 1] + table[i - 1][j - weight[i - 1]], table[i - 1][j]);
				}
			}
		}
		
		
		for(int i = 0; i <= weight.length; i++){
			for(int j  = 0; j <= capacity; j++){
				System.out.print(table[i][j] + " ");
			}
			System.out.println();
		}
	}

	private static int call(int i, int make){
		
		if(i >= weight.length)
			return 0;
		
		int profit1 = 0;
		int profit2 = 0;
		
		System.out.println(i + " " + make);
		if(capacity >= make + weight[i]){
			profit1 = profit[i] + call(i + 1, make + weight[i]);
		}
		profit2 = call(i + 1, make);
		int ans = Math.max(profit1, profit2);
		System.out.println(i + " " + make + " " + ans);
		System.out.println();
		return dp[i][make] = ans;
	}
}
