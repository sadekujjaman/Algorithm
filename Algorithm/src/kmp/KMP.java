package kmp;
import java.util.*;
 
public class KMP {
 
    static long INF = Long.MAX_VALUE;
 
   
    public static void main(String[] args) {
 
        Scanner sc = new Scanner(System.in);
       
//      String str1 = "abcdabex";
//      generatePrefixTable(str1);
//     
//      for(int i = 0; i < str1.length(); i++){
//          System.out.print(prefixTable[i] + " ");
//      }
        int test = sc.nextInt();
       
   
        for(int t = 1; t <= test; t++){
            String str = sc.next();
            String pattern = sc.next();
            Arrays.fill(prefixTable, 0);
            int ans = computeSubstring(str, pattern);
           
            System.out.println("Case " + t + ": " + ans);
        }
 
        sc.close();
 
    }
 
    private static int computeSubstring(String str, String pattern) {
       
        generatePrefixTable(pattern);
       
        int n = str.length();
        int m = pattern.length();
       
        int i = 0;
        int j = 0;
        int count = 0;
       
        while(i < n){
            if(str.charAt(i) == pattern.charAt(j)){
                if(j == m - 1){
                    count++;
//                  i = i - j + 1;
                    j = prefixTable[j - 1];
//                  i++;
                    continue;
                }
                i++;
                j++;
            }
            else if(j > 0){
                j = prefixTable[j - 1];
            }
            else{
             i++;
             
            }
        }
       
        return count;
    }
 
    static int[] prefixTable = new int[1000005];
   
    private static void generatePrefixTable(String pattern) {
        char[] patternArr = pattern.toCharArray();
        int n = pattern.length();
       
        int i = 1;
        int j = 0;
        prefixTable[0] = 0;
       
        while(i < n){
            if(patternArr[i] == patternArr[j]){
                prefixTable[i] = j + 1;
                i++;
                j++;
            }
            else if(j > 0){
                j = prefixTable[j - 1];
            }
            else{
                i++;
               
            }
        }
    }
 
}