import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Search, Filter, Eye, AlertTriangle, CheckCircle, Clock } from "lucide-react";

interface Claim {
  id: string;
  patientName: string;
  hospital: string;
  amount: string;
  date: string;
  fraudStatus: 'clean' | 'suspicious' | 'fraudulent' | 'pending';
  fraudReason?: string;
  location: string;
  claimType: string;
}

const mockClaims: Claim[] = [
  {
    id: 'CLM001',
    patientName: 'Rajesh Kumar',
    hospital: 'Apollo Hospital, Delhi',
    amount: '₹45,000',
    date: '2024-01-15',
    fraudStatus: 'clean',
    location: 'Delhi',
    claimType: 'Surgery'
  },
  {
    id: 'CLM002',
    patientName: 'Priya Sharma',
    hospital: 'Max Healthcare, Mumbai',
    amount: '₹1,25,000',
    date: '2024-01-14',
    fraudStatus: 'fraudulent',
    fraudReason: 'Overbilling - Amount exceeds hospital average by 300%',
    location: 'Mumbai',
    claimType: 'Emergency'
  },
  {
    id: 'CLM003',
    patientName: 'Amit Patel',
    hospital: 'Fortis Hospital, Bangalore',
    amount: '₹32,500',
    date: '2024-01-13',
    fraudStatus: 'suspicious',
    fraudReason: 'Duplicate claim pattern detected',
    location: 'Bangalore',
    claimType: 'Consultation'
  },
  {
    id: 'CLM004',
    patientName: 'Sunita Reddy',
    hospital: 'AIIMS, Delhi',
    amount: '₹67,800',
    date: '2024-01-12',
    fraudStatus: 'clean',
    location: 'Delhi',
    claimType: 'Treatment'
  },
  {
    id: 'CLM005',
    patientName: 'Vijay Singh',
    hospital: 'Unknown Clinic, Noida',
    amount: '₹89,000',
    date: '2024-01-11',
    fraudStatus: 'fraudulent',
    fraudReason: 'Hospital not verified in database',
    location: 'Noida',
    claimType: 'Surgery'
  }
];

export const ClaimsTable = () => {
  const [claims, setClaims] = useState<Claim[]>(mockClaims);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [locationFilter, setLocationFilter] = useState<string>('all');

  const filteredClaims = claims.filter(claim => {
    const matchesSearch = 
      claim.patientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      claim.hospital.toLowerCase().includes(searchTerm.toLowerCase()) ||
      claim.id.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || claim.fraudStatus === statusFilter;
    const matchesLocation = locationFilter === 'all' || claim.location === locationFilter;
    
    return matchesSearch && matchesStatus && matchesLocation;
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'clean':
        return <Badge className="bg-success text-success-foreground"><CheckCircle className="w-3 h-3 mr-1" />Clean</Badge>;
      case 'suspicious':
        return <Badge className="bg-warning text-warning-foreground"><AlertTriangle className="w-3 h-3 mr-1" />Suspicious</Badge>;
      case 'fraudulent':
        return <Badge className="bg-danger text-danger-foreground"><AlertTriangle className="w-3 h-3 mr-1" />Fraudulent</Badge>;
      case 'pending':
        return <Badge variant="secondary"><Clock className="w-3 h-3 mr-1" />Pending</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getAmountColor = (status: string) => {
    switch (status) {
      case 'fraudulent':
        return 'text-danger font-bold';
      case 'suspicious':
        return 'text-warning font-semibold';
      default:
        return 'text-foreground';
    }
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <Card className="bg-muted/20">
        <CardHeader>
          <CardTitle className="text-lg flex items-center">
            <Filter className="w-5 h-5 mr-2" />
            Filters & Search
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
              <Input
                placeholder="Search claims..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="clean">Clean</SelectItem>
                <SelectItem value="suspicious">Suspicious</SelectItem>
                <SelectItem value="fraudulent">Fraudulent</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
              </SelectContent>
            </Select>

            <Select value={locationFilter} onValueChange={setLocationFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by location" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Locations</SelectItem>
                <SelectItem value="Delhi">Delhi</SelectItem>
                <SelectItem value="Mumbai">Mumbai</SelectItem>
                <SelectItem value="Bangalore">Bangalore</SelectItem>
                <SelectItem value="Noida">Noida</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" onClick={() => {
              setSearchTerm('');
              setStatusFilter('all');
              setLocationFilter('all');
            }}>
              Clear Filters
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Claims Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            Claims Overview ({filteredClaims.length} results)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Claim ID</TableHead>
                  <TableHead>Patient Name</TableHead>
                  <TableHead>Hospital</TableHead>
                  <TableHead>Amount</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredClaims.map((claim) => (
                  <TableRow key={claim.id} className="hover:bg-muted/30">
                    <TableCell className="font-mono font-medium">{claim.id}</TableCell>
                    <TableCell className="font-medium">{claim.patientName}</TableCell>
                    <TableCell>
                      <div>
                        <div className="font-medium">{claim.hospital.split(',')[0]}</div>
                        <div className="text-sm text-muted-foreground">{claim.location}</div>
                      </div>
                    </TableCell>
                    <TableCell className={getAmountColor(claim.fraudStatus)}>
                      {claim.amount}
                    </TableCell>
                    <TableCell>{claim.date}</TableCell>
                    <TableCell>
                      <div className="space-y-1">
                        {getStatusBadge(claim.fraudStatus)}
                        {claim.fraudReason && (
                          <div className="text-xs text-muted-foreground max-w-xs">
                            {claim.fraudReason}
                          </div>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Button variant="outline" size="sm">
                        <Eye className="w-4 h-4 mr-1" />
                        View
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};