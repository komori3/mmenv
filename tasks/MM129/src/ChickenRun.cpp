// C++11
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  int dr[]={0,-1,0,1};
  int dc[]={-1,0,1,0};
  int elapsedTime;

  int N;
  cin >> N;

  vector<string> grid(N);
  for (int r=0; r<N; r++) {
    grid[r].resize(N);
    for (int c=0; c<N; c++) {
      cin >> grid[r][c];
      getchar();
    }
  }

  for (int turn=1; turn<=N*N; turn++)
  {
    vector<string> moves;
    vector<vector<bool>> used(N, vector<bool>(N, false));

    for (int r=0; r<N; r++)
      for (int c=0; c<N; c++)
        if (grid[r][c] == 'P' && !used[r][c])
          for (int m=0; m<4; m++)
          {
            int dir=(r*c+turn+m)%4;
            int r2=r+dr[dir];
            int c2=c+dc[dir];

            if (r2>=0 && r2<N && c2>=0 && c2<N && (grid[r2][c2]=='.' || grid[r2][c2]=='C'))
            {
              moves.push_back(to_string(r)+" "+to_string(c)+" "+to_string(r2)+" "+to_string(c2));
              grid[r][c]='.';
              grid[r2][c2]='P';
              used[r2][c2]=true;
              break;
            }
          }

    //print the moves
    cout << moves.size() << endl;
    for (string m : moves) cout << m << endl;
    cout.flush();

    //read elapsed time
    cin >> elapsedTime;

    //read the updated grid
    for (int r=0; r<N; r++) {
      for (int c=0; c<N; c++) {
        cin >> grid[r][c];
        getchar();
      }
    }
  }

  //signal end of solution
  cout << "-1";
  cout.flush();
}