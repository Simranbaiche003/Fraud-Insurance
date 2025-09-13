import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { BarChart3, PieChart, MapPin, TrendingDown, AlertTriangle } from "lucide-react";

interface AnalyticsDashboardProps {
  detailed?: boolean;
}

export const AnalyticsDashboard = ({ detailed = false }: AnalyticsDashboardProps) => {
  const regionData = [
    { region: 'Delhi', total: 847, fraudulent: 38, percentage: 4.5 },
    { region: 'Mumbai', total: 923, fraudulent: 52, percentage: 5.6 },
    { region: 'Bangalore', total: 654, fraudulent: 21, percentage: 3.2 },
    { region: 'Chennai', total: 423, fraudulent: 27, percentage: 6.4 },
  ];

  const hospitalData = [
    { hospital: 'Apollo Hospital', claims: 234, fraudulent: 12, amount: '₹2.4M' },
    { hospital: 'Max Healthcare', claims: 189, fraudulent: 18, amount: '₹3.1M' },
    { hospital: 'Fortis Hospital', claims: 167, fraudulent: 8, amount: '₹1.8M' },
    { hospital: 'AIIMS', claims: 145, fraudulent: 3, amount: '₹0.8M' },
    { hospital: 'Unknown Clinics', claims: 89, fraudulent: 67, amount: '₹8.9M' },
  ];

  const fraudTypes = [
    { type: 'Overbilling', count: 45, percentage: 35.4 },
    { type: 'Fake Hospital', count: 32, percentage: 25.2 },
    { type: 'Duplicate Claims', count: 28, percentage: 22.0 },
    { type: 'Document Forgery', count: 15, percentage: 11.8 },
    { type: 'Other', count: 7, percentage: 5.5 },
  ];

  return (
    <div className="space-y-6">
      {/* Quick Analytics Cards (always show) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Region-wise Fraud Heatmap */}
        <Card className="bg-gradient-to-br from-card to-card/50 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center text-lg">
              <MapPin className="w-5 h-5 mr-2 text-danger" />
              Regional Fraud Analysis
            </CardTitle>
            <CardDescription>Fraud distribution across regions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {regionData.map((region) => (
              <div key={region.region} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-medium">{region.region}</span>
                  <Badge 
                    className={
                      region.percentage > 5 
                        ? "bg-danger text-danger-foreground" 
                        : region.percentage > 3 
                        ? "bg-warning text-warning-foreground"
                        : "bg-success text-success-foreground"
                    }
                  >
                    {region.percentage}%
                  </Badge>
                </div>
                <Progress 
                  value={region.percentage} 
                  className="h-2" 
                />
                <div className="text-xs text-muted-foreground">
                  {region.fraudulent} of {region.total} claims
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Fraud Types Distribution */}
        <Card className="bg-gradient-to-br from-card to-card/50 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center text-lg">
              <PieChart className="w-5 h-5 mr-2 text-warning" />
              Fraud Types
            </CardTitle>
            <CardDescription>Distribution of fraud categories</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {fraudTypes.map((type) => (
              <div key={type.type} className="flex justify-between items-center">
                <div className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{
                      backgroundColor: type.type === 'Overbilling' ? 'hsl(var(--danger))' :
                                     type.type === 'Fake Hospital' ? 'hsl(var(--warning))' :
                                     type.type === 'Duplicate Claims' ? 'hsl(var(--info))' :
                                     'hsl(var(--muted-foreground))'
                    }}
                  />
                  <span className="text-sm font-medium">{type.type}</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-bold">{type.count}</div>
                  <div className="text-xs text-muted-foreground">{type.percentage}%</div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Monthly Trends */}
        <Card className="bg-gradient-to-br from-card to-card/50 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center text-lg">
              <TrendingDown className="w-5 h-5 mr-2 text-success" />
              Monthly Trends
            </CardTitle>
            <CardDescription>Fraud detection improvements</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm">Detection Accuracy</span>
                <span className="text-sm font-bold text-success">94.2%</span>
              </div>
              <Progress value={94.2} className="h-2" />
              
              <div className="flex justify-between items-center">
                <span className="text-sm">False Positives</span>
                <span className="text-sm font-bold text-warning">2.8%</span>
              </div>
              <Progress value={2.8} className="h-2" />
              
              <div className="flex justify-between items-center">
                <span className="text-sm">Processing Speed</span>
                <span className="text-sm font-bold text-info">98.5%</span>
              </div>
              <Progress value={98.5} className="h-2" />
            </div>
            
            <div className="pt-2 border-t border-border">
              <div className="text-xs text-muted-foreground">
                ↗️ 12% improvement this month
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analytics (only show when detailed prop is true) */}
      {detailed && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Hospital Analysis */}
          <Card className="bg-gradient-to-br from-card to-card/50 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center text-lg">
                <BarChart3 className="w-5 h-5 mr-2 text-info" />
                Hospital-wise Analysis
              </CardTitle>
              <CardDescription>Fraud patterns by healthcare providers</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {hospitalData.map((hospital) => (
                  <div key={hospital.hospital} className="p-3 bg-muted/30 rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h4 className="font-semibold">{hospital.hospital}</h4>
                        <p className="text-sm text-muted-foreground">
                          {hospital.claims} total claims
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-bold text-danger">
                          {hospital.fraudulent} fraudulent
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {hospital.amount} at risk
                        </div>
                      </div>
                    </div>
                    <Progress 
                      value={(hospital.fraudulent / hospital.claims) * 100} 
                      className="h-2"
                    />
                    <div className="flex justify-between items-center mt-1">
                      <span className="text-xs text-muted-foreground">
                        Risk Level: {
                          (hospital.fraudulent / hospital.claims) > 0.4 ? 'High' :
                          (hospital.fraudulent / hospital.claims) > 0.1 ? 'Medium' : 'Low'
                        }
                      </span>
                      <span className="text-xs font-medium">
                        {((hospital.fraudulent / hospital.claims) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Risk Indicators */}
          <Card className="bg-gradient-to-br from-card to-card/50 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center text-lg">
                <AlertTriangle className="w-5 h-5 mr-2 text-warning" />
                Risk Indicators
              </CardTitle>
              <CardDescription>Key fraud detection metrics and alerts</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-danger/10 rounded-lg border border-danger/20">
                  <div className="text-2xl font-bold text-danger">127</div>
                  <div className="text-sm text-danger">Active Alerts</div>
                </div>
                <div className="p-3 bg-warning/10 rounded-lg border border-warning/20">
                  <div className="text-2xl font-bold text-warning">89</div>
                  <div className="text-sm text-warning">Under Review</div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between p-2 bg-danger/5 rounded border-l-4 border-danger">
                  <div>
                    <div className="text-sm font-medium">High-Risk Pattern</div>
                    <div className="text-xs text-muted-foreground">
                      Unusual billing spike detected
                    </div>
                  </div>
                  <Badge className="bg-danger text-danger-foreground">Critical</Badge>
                </div>

                <div className="flex items-center justify-between p-2 bg-warning/5 rounded border-l-4 border-warning">
                  <div>
                    <div className="text-sm font-medium">Document Anomaly</div>
                    <div className="text-xs text-muted-foreground">
                      Suspicious document formatting
                    </div>
                  </div>
                  <Badge className="bg-warning text-warning-foreground">Medium</Badge>
                </div>

                <div className="flex items-center justify-between p-2 bg-info/5 rounded border-l-4 border-info">
                  <div>
                    <div className="text-sm font-medium">New Hospital</div>
                    <div className="text-xs text-muted-foreground">
                      Unverified healthcare provider
                    </div>
                  </div>
                  <Badge className="bg-info text-info-foreground">Low</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};