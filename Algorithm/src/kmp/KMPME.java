package kmp;

import java.io.*;
import java.util.*;
 
/**
 * 
 * @author Saju
 *
 */


public class KMPME {
 
    static long INF = Long.MAX_VALUE;
 
 
  static int[] prefixTable = new int[1000005];
   
    static String text = "";
    static String pattern = "";
   
    static int n = 0;
    public static void main(String[] args) {
 
    	InputReader in = new InputReader(System.in);
    	
        while(in.hasNext()){
            text = in.next();
            n = text.length();
            
            StringBuilder sb = new StringBuilder();
            sb.append(text);
            pattern = sb.reverse().toString();
            prefixTable = new int[n + 5];
            int index = kmpSearch();
//           System.out.println(index);
            for(int i = pattern.length()-1; i >= index; i--){
            	System.out.printf("%c", pattern.charAt(i));
            }
        	System.out.printf("%s\n", pattern);
//            System.gc();
        }
 
        
 
        System.exit(0);
    }
    
    private static void kmpPreprocess() {
       
        int i = 0, j = -1;
        prefixTable[i] = j;
        while (i < n) {
            while (j >= 0 && pattern.charAt(i) != pattern.charAt(j))
                j = prefixTable[j];
            i++;
            j++;
            prefixTable[i] = j;
        }
        
//        System.out.println(Arrays.toString(prefixTable));
    }
 
    private static int  kmpSearch() {
       
        kmpPreprocess();
       
       
        int i = 0, j = 0;
        while (i < n) {
            while (j >= 0 && text.charAt(i) != pattern.charAt(j))
                j = prefixTable[j];
            i++;
            j++;
        }
        return j;
    }
 
   

   
	
	static class InputReader{
		public BufferedReader reader;
		public StringTokenizer tokenizer;
		
		public InputReader(InputStream stream){
			reader = new BufferedReader(new InputStreamReader(stream));
			tokenizer = null;
		}
		
		public String next(){
			try{
				while(tokenizer == null || !tokenizer.hasMoreTokens()){
					tokenizer = new StringTokenizer(reader.readLine());
					
				}
			}
			catch(IOException e){
				return null;
			}
			return tokenizer.nextToken();
		}
		
		public String nextLine(){
			String line = null;
			try{
				tokenizer = null;
				line = reader.readLine();
			}
			catch(IOException e){
				throw new RuntimeException(e);
			}
			return line;
		}
		
		public int nextInt(){
			return Integer.parseInt(next());
		}
		public double nextDouble(){
			return Double.parseDouble(next());
		}
		public long nextLong(){
			return Long.parseLong(next());
		}
		public boolean hasNext(){
			try{
				while(tokenizer == null || !tokenizer.hasMoreTokens()){
					tokenizer = new StringTokenizer(reader.readLine());
				}
			}
			catch(Exception e){
				return false;
			}
			return true;
		}
	}
	
 }
 
