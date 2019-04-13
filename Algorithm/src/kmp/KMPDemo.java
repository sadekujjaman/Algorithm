package kmp;

public class KMPDemo {

	static int[] prefixTable = new int[102];
	 
	public static void main(String[] args) {
		
		String pattern = "ababaca";
		int len = pattern.length();
		generatePrefixTable(pattern.toCharArray(), len);
		
		for(int i = 0; i < len; i++){
			System.out.print(prefixTable[i] + " ");
		}
		System.out.println();
		
		
		String str = "bacbababababacaca";
		int index = patternMatchingKMP(str.toCharArray(), pattern.toCharArray());
		System.out.println(index);
	}
	
	static int patternMatchingKMP(char[] str, char[] pattern){
		
		int m = str.length;
		int n = pattern.length;
		int i = 0; 
		int j = 0;
		
		while(i < m){
			if(str[i] == pattern[j]){
				if(j == n - 1){
					System.out.println("pattern Exist");
					return i - j;
				}
				else{
					i++;
					j++;
				}
			}
			else if(j > 0){
				j = prefixTable[j - 1] ;
			}
			else{
				i++;
			}
		}
		System.out.println("pattern does not exist");
		return -1;
	}

	static void generatePrefixTable(char[] str, int n){
		int i = 1; 
		int j = 0;
		
		while(i < n){
			if(str[j] == str[i]){
				prefixTable[i] = j + 1;
				i++;
				j++;
			}
			else if(j > 0){
				j = prefixTable[j - 1];
			}
			else{
				prefixTable[j] = 0;
				i++;
			}
		}
	}
}
