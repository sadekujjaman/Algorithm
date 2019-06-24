package robincarp;

import java.util.Vector;

public class RobinCarpDemo {

	private static final int MAXCHAR = 256;
	private static final int MOD = 997;
	
	public static void main(String[] args) {
	
		
		
		String text = "abcdabcda";
		String pattern = "cda";
		
		int res = search(text, pattern);
		System.out.println(res);
		
	}

	private static int search(String text, String pattern) {
		int n = text.length();
		int m = pattern.length();
	
		
		int count = 0;
		
		int p = 53;
		
		long[] p_pow = new long[n];
		p_pow[0] = 1;
		
		for(int i = 1; i < n; i++){
			p_pow[i] = p_pow[i - 1] * p;
		}
		long [] h = new long[n];
		for(int i = 0; i < n; i++){
			
			h[i] = (text.charAt(i) - 'a' + 1) * p_pow[i];
			if(i > 0){
				h[i] = (h[i] + h[i - 1]);
			}
		}
		
		long h_s = 0;
		for(int i = 0; i < m; i++){
			h_s += (pattern.charAt(i) - 'a' + 1) * p_pow[i];//p_pow.get(i);
		}
		
		for(int i = 0; i + m - 1 < n; i++){
			long cur_h = h[i + m - 1];;
			if(i > 0){
				cur_h -= h[i - 1];
			}
			if(cur_h == h_s * p_pow[i]){
//				System.out.println("has");
				count++;
			}
		}
		
	
		return count;
	}
	
	

	private static boolean isEqual(char[] textArr, char[] patternArr, int start, int end) {
		int j = 0;
		for(int i = start; i <= end; i++){
			if(textArr[i] != patternArr[j++]){
				return false;
			}
		}
		return true;
	}

	private static boolean isEqual(String substring, String pattern, int m) {
		for(int i = 0; i < m; i++ ){
			if(substring.charAt(i) != pattern.charAt(i)){
				return false;
			}
		}
		return true;
	}

	private static long pow(int base, int pow, int mod) {
		if(pow == 0){
			return 1;
		}
		
		if(pow % 2 == 0){
			return pow((base * base) % mod, pow / 2, mod);
		}
		else{
			return (base * pow((base * base) % mod, pow / 2, mod)) % mod;
		}
		
	}

}
